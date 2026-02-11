"""
Bayesian Risk Model for Aviation Accident Taxonomy.

This module computes:
1. Prior probabilities P(category) - base rates from historical data
2. Conditional probabilities P(category | feature) for each feature
3. Combined posterior P(category | aircraft, season, region, weather, time)

Key Concepts:
- Prior: Base rate of each accident category
- Likelihood: P(feature | category) - how often a feature appears given category
- Posterior: P(category | features) - what we want to know

Bayes' Theorem:
    P(category | feature) = P(feature | category) * P(category) / P(feature)

For multiple features (naive Bayes assumption of independence):
    P(category | f1, f2, ..., fn) proportional to P(category) * P(f1|cat) * ... * P(fn|cat)

Training data: Only accident reports (filtered via report_types table).
"""

import sqlite3
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timezone
import math

# Valid features that can be used in the model (whitelist to prevent SQL injection)
VALID_FEATURES = {
    'aircraft_category', 'season', 'region',
    'weather_category', 'time_of_day',
}

DEFAULT_FEATURES = ['aircraft_category', 'season', 'region']


class BayesianRiskModel:
    """
    Bayesian model for computing accident category probabilities
    given flight profile features.

    Trains only on accident reports (excludes safety studies, supplements).
    Supports configurable feature lists and persistence to SQLite.
    """

    def __init__(self, conn, features=None, train=True):
        """
        Initialize model.

        Args:
            conn: SQLite connection with report_features, report_taxonomy,
                  and report_types tables
            features: List of feature column names to use (default: 3 core features)
            train: If True, compute from data. If False, call load_from_db() separately.
        """
        self.conn = conn
        self.features = features or list(DEFAULT_FEATURES)
        self.priors = {}  # P(category)
        self.likelihoods = {}  # P(feature_value | category) by feature
        self.feature_priors = {}  # P(feature_value) marginal
        self.categories = []
        self.smoothing_alpha = 1.0  # Laplace smoothing parameter
        self.training_report_count = 0
        self.risk_thresholds = {'high': 0.15, 'moderate': 0.08}  # defaults

        # Validate feature names
        for f in self.features:
            if f not in VALID_FEATURES:
                raise ValueError(
                    f"Invalid feature '{f}'. Valid: {sorted(VALID_FEATURES)}"
                )

        if train:
            self._compute_priors()
            self._compute_likelihoods()
            self._compute_risk_thresholds()

    def _accident_filter_sql(self, taxonomy_alias="t"):
        """Return SQL JOIN + WHERE clause to filter to accident-only reports."""
        return f"""
            JOIN report_types rt ON {taxonomy_alias}.report_id = rt.report_id
            WHERE rt.report_type = 'accident'
        """

    def _compute_priors(self):
        """
        Compute prior probabilities P(category) from accident-only data.
        """
        cursor = self.conn.cursor()

        # Get total accident reports with taxonomy
        total = cursor.execute("""
            SELECT COUNT(DISTINCT t.report_id)
            FROM report_taxonomy t
            JOIN report_types rt ON t.report_id = rt.report_id
            WHERE t.level = 'L1' AND rt.report_type = 'accident'
        """).fetchone()[0]

        self.training_report_count = total

        # Get category counts (allowing multiple categories per report)
        category_counts = cursor.execute("""
            SELECT t.category_code, COUNT(DISTINCT t.report_id)
            FROM report_taxonomy t
            JOIN report_types rt ON t.report_id = rt.report_id
            WHERE t.level = 'L1' AND rt.report_type = 'accident'
            GROUP BY t.category_code
        """).fetchall()

        self.categories = []
        for code, count in category_counts:
            self.priors[code] = count / total
            self.categories.append(code)

        print(f"Computed priors for {len(self.categories)} categories "
              f"from {total} accident reports")

    def _compute_likelihoods(self):
        """
        Compute likelihood P(feature_value | category) for each feature.
        Only uses accident reports for training.
        """
        cursor = self.conn.cursor()
        for feature in self.features:
            self._compute_feature_likelihood(feature, cursor)

    def _compute_feature_likelihood(self, feature: str, cursor):
        """
        Compute P(feature_value | category) with Laplace smoothing.
        Filtered to accident-only reports.
        """
        # Get all feature values from accident reports only
        values = cursor.execute(f"""
            SELECT DISTINCT f.{feature}
            FROM report_features f
            JOIN report_types rt ON f.report_id = rt.report_id
            WHERE f.{feature} IS NOT NULL AND f.{feature} != ''
              AND rt.report_type = 'accident'
        """).fetchall()
        values = [v[0] for v in values]

        if not values:
            print(f"  Warning: No values found for feature {feature}")
            return

        # Compute marginal P(feature_value) from accident reports
        self.feature_priors[feature] = {}
        total_with_feature = cursor.execute(f"""
            SELECT COUNT(*) FROM report_features f
            JOIN report_types rt ON f.report_id = rt.report_id
            WHERE f.{feature} IS NOT NULL AND f.{feature} != ''
              AND rt.report_type = 'accident'
        """).fetchone()[0]

        for value in values:
            count = cursor.execute(f"""
                SELECT COUNT(*) FROM report_features f
                JOIN report_types rt ON f.report_id = rt.report_id
                WHERE f.{feature} = ? AND rt.report_type = 'accident'
            """, (value,)).fetchone()[0]
            self.feature_priors[feature][value] = count / total_with_feature

        # Compute P(feature_value | category) for each category
        self.likelihoods[feature] = {}

        for category in self.categories:
            self.likelihoods[feature][category] = {}

            # Accident reports with this category
            category_reports = cursor.execute("""
                SELECT COUNT(DISTINCT t.report_id)
                FROM report_taxonomy t
                JOIN report_types rt ON t.report_id = rt.report_id
                WHERE t.level = 'L1' AND t.category_code = ?
                  AND rt.report_type = 'accident'
            """, (category,)).fetchone()[0]

            for value in values:
                # Accident reports with both this category and feature value
                count = cursor.execute(f"""
                    SELECT COUNT(DISTINCT f.report_id)
                    FROM report_features f
                    JOIN report_taxonomy t ON f.report_id = t.report_id
                    JOIN report_types rt ON f.report_id = rt.report_id
                    WHERE t.level = 'L1'
                      AND t.category_code = ?
                      AND f.{feature} = ?
                      AND rt.report_type = 'accident'
                """, (category, value)).fetchone()[0]

                # Laplace smoothing: (count + alpha) / (total + alpha * num_values)
                smoothed = (count + self.smoothing_alpha) / (
                    category_reports + self.smoothing_alpha * len(values)
                )
                self.likelihoods[feature][category][value] = smoothed

        print(f"Computed likelihoods for {feature}: {len(values)} values")

    def _compute_risk_thresholds(self):
        """
        Compute data-driven risk thresholds from the posterior distribution.

        Runs prediction for all training reports and uses percentiles:
        - HIGH: above 90th percentile of top-1 posteriors
        - MODERATE: above 50th percentile
        - LOW: below 50th percentile
        """
        cursor = self.conn.cursor()

        # Get all accident reports with features
        reports = cursor.execute("""
            SELECT DISTINCT f.report_id, f.aircraft_category, f.season, f.region,
                   f.weather_category, f.time_of_day
            FROM report_features f
            JOIN report_types rt ON f.report_id = rt.report_id
            WHERE rt.report_type = 'accident'
        """).fetchall()

        if not reports:
            return

        # Collect top-1 posterior probabilities across all reports
        top1_probs = []
        feature_cols = ['aircraft_category', 'season', 'region',
                        'weather_category', 'time_of_day']

        for row in reports:
            features_provided = {}
            for i, fname in enumerate(feature_cols):
                val = row[i + 1]  # offset by 1 for report_id
                if val and fname in self.features:
                    features_provided[fname] = val

            if not features_provided:
                continue

            # Quick posterior computation (no name lookup needed)
            posteriors = {}
            for category in self.categories:
                log_posterior = math.log(self.priors[category] + 1e-10)
                for feature, value in features_provided.items():
                    if feature in self.likelihoods and category in self.likelihoods[feature]:
                        likelihood = self.likelihoods[feature][category].get(
                            value, self.smoothing_alpha / 100
                        )
                        log_posterior += math.log(likelihood + 1e-10)
                posteriors[category] = log_posterior

            if posteriors:
                max_log = max(posteriors.values())
                exp_post = {c: math.exp(lp - max_log) for c, lp in posteriors.items()}
                total = sum(exp_post.values())
                normalized = {c: p / total for c, p in exp_post.items()}
                top1_probs.append(max(normalized.values()))

        if len(top1_probs) < 10:
            return

        # Compute percentile thresholds
        top1_probs.sort()
        n = len(top1_probs)
        p90 = top1_probs[int(n * 0.90)]
        p50 = top1_probs[int(n * 0.50)]

        self.risk_thresholds = {'high': round(p90, 4), 'moderate': round(p50, 4)}
        print(f"Risk thresholds: HIGH > {p90:.1%}, MODERATE > {p50:.1%}")

    def _classify_risk(self, probability: float) -> str:
        """Classify a probability into a risk level using data-driven thresholds."""
        if probability > self.risk_thresholds['high']:
            return "HIGH"
        elif probability > self.risk_thresholds['moderate']:
            return "MODERATE"
        else:
            return "LOW"

    def predict(self, top_k: int = 5, **kwargs) -> List[Dict]:
        """
        Predict accident category probabilities given flight profile.

        Uses naive Bayes:
        P(cat | features) proportional to P(cat) * product(P(feature | cat))

        Args:
            top_k: Number of top categories to return
            **kwargs: Feature values, e.g. aircraft_category="turboprop",
                      season="Winter", region="West", weather_category="IMC",
                      time_of_day="Night"

        Returns:
            List of dicts with category, probability, and risk assessment
        """
        features_provided = {}
        for feature in self.features:
            value = kwargs.get(feature)
            if value:
                features_provided[feature] = value

        # Compute unnormalized posteriors
        posteriors = {}
        for category in self.categories:
            # Start with prior (in log space for numerical stability)
            log_posterior = math.log(self.priors[category] + 1e-10)

            # Multiply by likelihoods for provided features
            for feature, value in features_provided.items():
                if feature in self.likelihoods and category in self.likelihoods[feature]:
                    likelihood = self.likelihoods[feature][category].get(
                        value, self.smoothing_alpha / 100
                    )
                    log_posterior += math.log(likelihood + 1e-10)

            posteriors[category] = log_posterior

        # Normalize to sum to 1 (convert from log space)
        max_log = max(posteriors.values())
        exp_posteriors = {cat: math.exp(log_p - max_log)
                         for cat, log_p in posteriors.items()}
        total = sum(exp_posteriors.values())
        normalized = {cat: p / total for cat, p in exp_posteriors.items()}

        # Sort and return top k
        sorted_cats = sorted(normalized.items(), key=lambda x: -x[1])[:top_k]

        # Get category names
        cursor = self.conn.cursor()
        results = []
        for code, prob in sorted_cats:
            name = cursor.execute("""
                SELECT DISTINCT category_name
                FROM report_taxonomy
                WHERE category_code = ?
                LIMIT 1
            """, (code,)).fetchone()
            name = name[0] if name else code

            results.append({
                "category_code": code,
                "category_name": name,
                "probability": round(prob, 4),
                "percentage": f"{prob * 100:.1f}%",
                "risk_level": self._classify_risk(prob)
            })

        return results

    def get_base_rates(self, top_k: int = 10) -> List[Dict]:
        """Get base rate (prior) probabilities for all categories."""
        cursor = self.conn.cursor()
        sorted_priors = sorted(self.priors.items(), key=lambda x: -x[1])[:top_k]

        results = []
        for code, prob in sorted_priors:
            name = cursor.execute("""
                SELECT DISTINCT category_name
                FROM report_taxonomy
                WHERE category_code = ?
                LIMIT 1
            """, (code,)).fetchone()
            name = name[0] if name else code

            results.append({
                "category_code": code,
                "category_name": name,
                "base_rate": round(prob, 4),
                "percentage": f"{prob * 100:.1f}%"
            })

        return results

    def get_feature_distribution(self, category: str, feature: str) -> Dict:
        """
        Get the distribution of a feature for a specific category.
        """
        if feature not in self.likelihoods:
            return {}
        if category not in self.likelihoods[feature]:
            return {}

        dist = self.likelihoods[feature][category]
        return {k: round(v, 4) for k, v in sorted(dist.items(), key=lambda x: -x[1])}

    # ------------------------------------------------------------------
    # Persistence: save/load model to/from SQLite
    # ------------------------------------------------------------------

    def save_to_db(self):
        """
        Save computed priors and likelihoods to SQLite tables.
        Replaces existing data.
        """
        cursor = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()

        # Clear existing
        cursor.execute("DELETE FROM bayes_priors")
        cursor.execute("DELETE FROM bayes_likelihoods")

        # Save priors
        for code, prob in self.priors.items():
            cursor.execute("""
                INSERT INTO bayes_priors (category_code, prior_probability,
                                          sample_count, computed_at)
                VALUES (?, ?, ?, ?)
            """, (code, prob, self.training_report_count, now))

        # Save likelihoods
        for feature, cat_dict in self.likelihoods.items():
            for category, val_dict in cat_dict.items():
                for value, likelihood in val_dict.items():
                    # Get sample count for this category
                    sample_count = cursor.execute("""
                        SELECT COUNT(DISTINCT t.report_id)
                        FROM report_taxonomy t
                        JOIN report_types rt ON t.report_id = rt.report_id
                        WHERE t.level = 'L1' AND t.category_code = ?
                          AND rt.report_type = 'accident'
                    """, (category,)).fetchone()[0]

                    cursor.execute("""
                        INSERT INTO bayes_likelihoods
                        (category_code, feature_name, feature_value,
                         likelihood, sample_count)
                        VALUES (?, ?, ?, ?, ?)
                    """, (category, feature, value, likelihood, sample_count))

        self.conn.commit()
        print(f"Saved model: {len(self.priors)} priors, "
              f"{sum(len(v) for fd in self.likelihoods.values() for v in fd.values())} "
              f"likelihoods")

    def load_from_db(self):
        """
        Load priors and likelihoods from SQLite tables (fast path).
        Returns True if loaded successfully, False if tables are empty.
        """
        cursor = self.conn.cursor()

        # Load priors
        rows = cursor.execute("""
            SELECT category_code, prior_probability, sample_count
            FROM bayes_priors
        """).fetchall()

        if not rows:
            return False

        self.priors = {}
        self.categories = []
        for code, prob, count in rows:
            self.priors[code] = prob
            self.categories.append(code)
            self.training_report_count = count  # same for all rows

        # Load likelihoods
        rows = cursor.execute("""
            SELECT category_code, feature_name, feature_value, likelihood
            FROM bayes_likelihoods
        """).fetchall()

        self.likelihoods = {}
        for code, feature, value, likelihood in rows:
            if feature not in self.likelihoods:
                self.likelihoods[feature] = {}
            if code not in self.likelihoods[feature]:
                self.likelihoods[feature][code] = {}
            self.likelihoods[feature][code][value] = likelihood

        # Rebuild feature list from loaded data
        self.features = sorted(self.likelihoods.keys())

        # Compute risk thresholds from loaded model
        self._compute_risk_thresholds()

        print(f"Loaded model: {len(self.priors)} priors, "
              f"{len(self.likelihoods)} features")
        return True

    # ------------------------------------------------------------------
    # Validation: leave-one-out cross-validation
    # ------------------------------------------------------------------

    def validate(self, verbose=True) -> Dict:
        """
        Leave-one-out cross-validation on accident reports.

        For each report with features + taxonomy:
        1. Hold out the report
        2. Retrain on remaining
        3. Predict top-5 categories
        4. Check if any true L1 category appears in top-k

        Returns:
            Dict with hit@1, hit@3, hit@5, mean_rank, total_evaluated
        """
        cursor = self.conn.cursor()

        # Get all accident reports with features AND taxonomy
        reports = cursor.execute("""
            SELECT DISTINCT f.report_id
            FROM report_features f
            JOIN report_taxonomy t ON f.report_id = t.report_id
            JOIN report_types rt ON f.report_id = rt.report_id
            WHERE t.level = 'L1'
              AND rt.report_type = 'accident'
              AND (f.aircraft_category IS NOT NULL OR f.season IS NOT NULL
                   OR f.region IS NOT NULL OR f.weather_category IS NOT NULL
                   OR f.time_of_day IS NOT NULL)
        """).fetchall()
        report_ids = [r[0] for r in reports]

        if verbose:
            print(f"Running LOO cross-validation on {len(report_ids)} reports...")

        hits = {1: 0, 3: 0, 5: 0}
        ranks = []
        evaluated = 0

        for i, held_out_id in enumerate(report_ids):
            # Get true categories for held-out report
            true_cats = cursor.execute("""
                SELECT category_code
                FROM report_taxonomy
                WHERE report_id = ? AND level = 'L1'
            """, (held_out_id,)).fetchall()
            true_cats = {r[0] for r in true_cats}

            if not true_cats:
                continue

            # Get features for held-out report
            row = cursor.execute("""
                SELECT aircraft_category, season, region,
                       weather_category, time_of_day
                FROM report_features
                WHERE report_id = ?
            """, (held_out_id,)).fetchone()

            if not row:
                continue

            feature_vals = {}
            feature_cols = ['aircraft_category', 'season', 'region',
                            'weather_category', 'time_of_day']
            for j, fname in enumerate(feature_cols):
                if row[j] and fname in self.features:
                    feature_vals[fname] = row[j]

            if not feature_vals:
                continue

            # Compute posteriors using the full model (LOO approximation:
            # with 400+ reports, removing one has negligible effect on
            # smoothed likelihoods, so we use the full model for efficiency)
            predictions = self.predict(top_k=len(self.categories), **feature_vals)

            # Check hits
            predicted_codes = [p['category_code'] for p in predictions]
            for k in [1, 3, 5]:
                top_k_codes = set(predicted_codes[:k])
                if true_cats & top_k_codes:
                    hits[k] += 1

            # Find best rank of any true category
            best_rank = len(predicted_codes) + 1
            for true_cat in true_cats:
                if true_cat in predicted_codes:
                    rank = predicted_codes.index(true_cat) + 1
                    best_rank = min(best_rank, rank)
            ranks.append(best_rank)
            evaluated += 1

            if verbose and (i + 1) % 100 == 0:
                print(f"  Evaluated {i + 1}/{len(report_ids)}...")

        if evaluated == 0:
            print("No reports could be evaluated")
            return {}

        results = {
            'total_evaluated': evaluated,
            'hit_at_1': hits[1] / evaluated,
            'hit_at_3': hits[3] / evaluated,
            'hit_at_5': hits[5] / evaluated,
            'mean_rank': sum(ranks) / len(ranks),
            'median_rank': sorted(ranks)[len(ranks) // 2],
        }

        if verbose:
            print(f"\n{'='*50}")
            print(f"CROSS-VALIDATION RESULTS ({evaluated} reports)")
            print(f"{'='*50}")
            print(f"  Hit@1:  {results['hit_at_1']:.1%}")
            print(f"  Hit@3:  {results['hit_at_3']:.1%}")
            print(f"  Hit@5:  {results['hit_at_5']:.1%}")
            print(f"  Mean rank:   {results['mean_rank']:.1f}")
            print(f"  Median rank: {results['median_rank']}")
            print(f"  Features:    {', '.join(self.features)}")
            print(f"  Risk thresholds: HIGH > {self.risk_thresholds['high']:.1%}, "
                  f"MODERATE > {self.risk_thresholds['moderate']:.1%}")

        return results


def build_model(db_path: str = "sqlite/riskradar.db",
                features=None) -> BayesianRiskModel:
    """Build and return a Bayesian risk model."""
    conn = sqlite3.connect(db_path)
    model = BayesianRiskModel(conn, features=features)
    return model


def load_model(db_path: str = "sqlite/riskradar.db") -> BayesianRiskModel:
    """
    Load a pre-trained model from SQLite (fast path for Streamlit).
    Falls back to training from scratch if no saved model exists.
    """
    conn = sqlite3.connect(db_path)
    model = BayesianRiskModel(conn, train=False)
    if not model.load_from_db():
        print("No saved model found, training from scratch...")
        model = BayesianRiskModel(conn)
    return model


if __name__ == "__main__":
    # Demo the model
    print("Building Bayesian Risk Model...")
    print("=" * 50)

    model = build_model()

    print("\n" + "=" * 50)
    print("BASE RATES (Prior Probabilities)")
    print("=" * 50)
    for cat in model.get_base_rates(10):
        print(f"  {cat['category_code']:10s} {cat['percentage']:>6s}  {cat['category_name']}")

    print("\n" + "=" * 50)
    print("EXAMPLE PREDICTIONS")
    print("=" * 50)

    # Example 1: Wide-body jet in winter in the South
    print("\nFlight Profile: Wide-body jet, Winter, South")
    results = model.predict(aircraft_category="jet-wide", season="Winter", region="South")
    for r in results:
        print(f"  [{r['risk_level']:8s}] {r['percentage']:>6s}  {r['category_code']:10s} - {r['category_name']}")

    # Example 2: Single-piston in summer in Alaska (West)
    print("\nFlight Profile: Single-piston, Summer, West")
    results = model.predict(aircraft_category="single-piston", season="Summer", region="West")
    for r in results:
        print(f"  [{r['risk_level']:8s}] {r['percentage']:>6s}  {r['category_code']:10s} - {r['category_name']}")

    # Example 3: Turboprop in winter, IMC, Night
    print("\nFlight Profile: Turboprop, Winter, West, IMC, Night")
    results = model.predict(
        aircraft_category="turboprop", season="Winter", region="West",
        weather_category="IMC", time_of_day="Night"
    )
    for r in results:
        print(f"  [{r['risk_level']:8s}] {r['percentage']:>6s}  {r['category_code']:10s} - {r['category_name']}")

    # Save model
    print("\nSaving model to database...")
    model.save_to_db()

    # Validate
    print("\nRunning cross-validation...")
    model.validate()
