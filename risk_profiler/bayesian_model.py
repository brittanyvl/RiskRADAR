"""
Bayesian Risk Model for Aviation Accident Taxonomy.

This module computes:
1. Prior probabilities P(category) - base rates from historical data
2. Conditional probabilities P(category | feature) for each feature
3. Combined posterior P(category | aircraft, season, region)

Key Concepts:
- Prior: Base rate of each accident category
- Likelihood: P(feature | category) - how often a feature appears given category
- Posterior: P(category | features) - what we want to know

Bayes' Theorem:
    P(category | feature) = P(feature | category) * P(category) / P(feature)

For multiple features (naive Bayes assumption of independence):
    P(category | f1, f2, f3) proportional to P(category) * P(f1|cat) * P(f2|cat) * P(f3|cat)
"""

import sqlite3
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import math


class BayesianRiskModel:
    """
    Bayesian model for computing accident category probabilities
    given flight profile features.
    """

    def __init__(self, conn):
        """
        Initialize model from database.

        Args:
            conn: SQLite connection with report_features and report_taxonomy tables
        """
        self.conn = conn
        self.priors = {}  # P(category)
        self.likelihoods = {}  # P(feature_value | category) by feature
        self.feature_priors = {}  # P(feature_value) marginal
        self.categories = []
        self.smoothing_alpha = 1.0  # Laplace smoothing parameter

        self._compute_priors()
        self._compute_likelihoods()

    def _compute_priors(self):
        """
        Compute prior probabilities P(category) from data.

        Uses L1 category assignments with weighted confidence.
        """
        cursor = self.conn.cursor()

        # Get total reports with taxonomy
        total = cursor.execute("""
            SELECT COUNT(DISTINCT report_id)
            FROM report_taxonomy
            WHERE level = 'L1'
        """).fetchone()[0]

        # Get category counts (allowing multiple categories per report)
        category_counts = cursor.execute("""
            SELECT category_code, COUNT(DISTINCT report_id)
            FROM report_taxonomy
            WHERE level = 'L1'
            GROUP BY category_code
        """).fetchall()

        # Compute priors as proportion of reports with each category
        self.categories = []
        total_assignments = sum(count for _, count in category_counts)

        for code, count in category_counts:
            # Use proportion of assignments (can sum to >1 due to multi-label)
            self.priors[code] = count / total
            self.categories.append(code)

        print(f"Computed priors for {len(self.categories)} categories from {total} reports")

    def _compute_likelihoods(self):
        """
        Compute likelihood P(feature_value | category) for each feature.

        Features computed:
        - aircraft_category: jet-wide, jet-narrow, turboprop, etc.
        - season: Winter, Spring, Summer, Fall
        - region: Northeast, South, Midwest, West
        """
        cursor = self.conn.cursor()

        # Features to compute
        features = ['aircraft_category', 'season', 'region']

        for feature in features:
            self._compute_feature_likelihood(feature, cursor)

    def _compute_feature_likelihood(self, feature: str, cursor):
        """
        Compute P(feature_value | category) with Laplace smoothing.
        """
        # Get all feature values
        values = cursor.execute(f"""
            SELECT DISTINCT {feature}
            FROM report_features
            WHERE {feature} IS NOT NULL AND {feature} != ''
        """).fetchall()
        values = [v[0] for v in values]

        if not values:
            print(f"  Warning: No values found for feature {feature}")
            return

        # Compute marginal P(feature_value)
        self.feature_priors[feature] = {}
        total_with_feature = cursor.execute(f"""
            SELECT COUNT(*) FROM report_features
            WHERE {feature} IS NOT NULL AND {feature} != ''
        """).fetchone()[0]

        for value in values:
            count = cursor.execute(f"""
                SELECT COUNT(*) FROM report_features
                WHERE {feature} = ?
            """, (value,)).fetchone()[0]
            self.feature_priors[feature][value] = count / total_with_feature

        # Compute P(feature_value | category) for each category
        self.likelihoods[feature] = {}

        for category in self.categories:
            self.likelihoods[feature][category] = {}

            # Reports with this category
            category_reports = cursor.execute("""
                SELECT COUNT(DISTINCT report_id)
                FROM report_taxonomy
                WHERE level = 'L1' AND category_code = ?
            """, (category,)).fetchone()[0]

            for value in values:
                # Reports with both this category and this feature value
                count = cursor.execute(f"""
                    SELECT COUNT(DISTINCT f.report_id)
                    FROM report_features f
                    JOIN report_taxonomy t ON f.report_id = t.report_id
                    WHERE t.level = 'L1'
                      AND t.category_code = ?
                      AND f.{feature} = ?
                """, (category, value)).fetchone()[0]

                # Laplace smoothing: (count + alpha) / (total + alpha * num_values)
                smoothed = (count + self.smoothing_alpha) / (
                    category_reports + self.smoothing_alpha * len(values)
                )
                self.likelihoods[feature][category][value] = smoothed

        print(f"Computed likelihoods for {feature}: {len(values)} values")

    def predict(
        self,
        aircraft_category: Optional[str] = None,
        season: Optional[str] = None,
        region: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Predict accident category probabilities given flight profile.

        Uses naive Bayes:
        P(cat | features) proportional to P(cat) * product(P(feature | cat))

        Args:
            aircraft_category: e.g., "jet-narrow", "turboprop"
            season: e.g., "Winter", "Summer"
            region: e.g., "South", "West"
            top_k: Number of top categories to return

        Returns:
            List of dicts with category, probability, and risk assessment
        """
        features_provided = {}
        if aircraft_category:
            features_provided['aircraft_category'] = aircraft_category
        if season:
            features_provided['season'] = season
        if region:
            features_provided['region'] = region

        # Compute unnormalized posteriors
        posteriors = {}
        for category in self.categories:
            # Start with prior (in log space for numerical stability)
            log_posterior = math.log(self.priors[category] + 1e-10)

            # Multiply by likelihoods for provided features
            for feature, value in features_provided.items():
                if feature in self.likelihoods and category in self.likelihoods[feature]:
                    likelihood = self.likelihoods[feature][category].get(value, self.smoothing_alpha / 100)
                    log_posterior += math.log(likelihood + 1e-10)

            posteriors[category] = log_posterior

        # Normalize to sum to 1 (convert from log space)
        max_log = max(posteriors.values())
        exp_posteriors = {cat: math.exp(log_p - max_log) for cat, log_p in posteriors.items()}
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

            # Risk assessment
            if prob > 0.15:
                risk_level = "HIGH"
            elif prob > 0.08:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"

            results.append({
                "category_code": code,
                "category_name": name,
                "probability": round(prob, 4),
                "percentage": f"{prob * 100:.1f}%",
                "risk_level": risk_level
            })

        return results

    def get_base_rates(self, top_k: int = 10) -> List[Dict]:
        """
        Get base rate (prior) probabilities for all categories.
        """
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

        Useful for understanding which feature values are associated
        with which categories.
        """
        if feature not in self.likelihoods:
            return {}
        if category not in self.likelihoods[feature]:
            return {}

        dist = self.likelihoods[feature][category]
        return {k: round(v, 4) for k, v in sorted(dist.items(), key=lambda x: -x[1])}


def build_model(db_path: str = "sqlite/riskradar.db") -> BayesianRiskModel:
    """
    Build and return a Bayesian risk model.
    """
    conn = sqlite3.connect(db_path)
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

    # Example 3: Turboprop with no other info
    print("\nFlight Profile: Turboprop (no other info)")
    results = model.predict(aircraft_category="turboprop")
    for r in results:
        print(f"  [{r['risk_level']:8s}] {r['percentage']:>6s}  {r['category_code']:10s} - {r['category_name']}")
