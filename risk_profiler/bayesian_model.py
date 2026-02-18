"""
Binary Relevance Naive Bayes Risk Model for Aviation Accident Taxonomy.

This module uses Binary Relevance — 27 independent binary classifiers,
one per CICTT L1 category. Each classifier computes:

    P(c=1 | features) = sigmoid(log P(c=1) + Σ log P(fi|c=1)
                                - log P(c=0) - Σ log P(fi|c=0))

Key advantages over single-label softmax:
- Multiple categories CAN have high probability simultaneously (multi-label)
- No cross-category normalization (probabilities don't sum to 1)
- Proper calibration (predicted probability ≈ actual frequency)
- Eliminates 3-4x systematic calibration error from softmax on multi-label data

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


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


class BayesianRiskModel:
    """
    Binary Relevance Naive Bayes model for computing accident category
    probabilities given flight profile features.

    Each of 27 CICTT categories gets an independent binary classifier.
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
        self.priors = {}  # P(c=1) for each category
        self.likelihoods = {}  # {feat: {cat: {val: (lik_pos, lik_neg)}}}
        self.categories = []
        self.smoothing_alpha = 1.0  # Laplace smoothing parameter
        self.training_report_count = 0
        self.risk_thresholds = {'high': 0.15, 'moderate': 0.08}  # defaults

        # Binary relevance specific
        self.category_report_counts = {}  # {cat: (pos_count, neg_count)}
        self.feature_value_sets = {}  # {feat: [sorted values]}

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

    def _compute_priors(self):
        """
        Compute prior probabilities P(c=1) from accident-only data.
        P(c=1) = count of reports with category / total accident reports.
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
            self.category_report_counts[code] = (count, total - count)

        print(f"Computed priors for {len(self.categories)} categories "
              f"from {total} accident reports")

    def _compute_likelihoods(self):
        """
        Compute likelihoods P(feature_value | c=1) and P(feature_value | c=0)
        for each feature. Only uses accident reports for training.
        """
        cursor = self.conn.cursor()
        for feature in self.features:
            self._compute_feature_likelihood(feature, cursor)

    def _compute_feature_likelihood(self, feature: str, cursor):
        """
        Compute P(val|c=1) and P(val|c=0) with Laplace smoothing.

        Optimized: 2 bulk queries per feature instead of O(categories × values).
        """
        alpha = self.smoothing_alpha

        # 1. Get all (report_id, feature_value) pairs for accident reports
        report_values = cursor.execute(f"""
            SELECT f.report_id, f.{feature}
            FROM report_features f
            JOIN report_types rt ON f.report_id = rt.report_id
            WHERE f.{feature} IS NOT NULL AND f.{feature} != ''
              AND rt.report_type = 'accident'
        """).fetchall()

        if not report_values:
            print(f"  Warning: No values found for feature {feature}")
            return

        # Build report -> value lookup
        report_to_value = {}
        values_set = set()
        for report_id, value in report_values:
            report_to_value[report_id] = value
            values_set.add(value)

        values = sorted(values_set)
        self.feature_value_sets[feature] = values
        n_values = len(values)

        # 2. Get all (report_id, category_code) pairs for accident reports
        report_cats = cursor.execute("""
            SELECT t.report_id, t.category_code
            FROM report_taxonomy t
            JOIN report_types rt ON t.report_id = rt.report_id
            WHERE t.level = 'L1' AND rt.report_type = 'accident'
        """).fetchall()

        # Build report -> set of categories lookup
        report_to_cats = defaultdict(set)
        for report_id, cat_code in report_cats:
            report_to_cats[report_id].add(cat_code)

        # 3. Count (value, category) co-occurrences for pos and neg
        # pos_counts[cat][val] = # reports with cat AND val
        # neg_counts[cat][val] = # reports WITHOUT cat AND val
        pos_counts = {cat: defaultdict(int) for cat in self.categories}
        neg_counts = {cat: defaultdict(int) for cat in self.categories}

        # Also track total reports with feature per class
        # cat_pos_with_feature[cat] = # reports with cat that have this feature
        # cat_neg_with_feature[cat] = # reports without cat that have this feature
        cat_pos_with_feature = defaultdict(int)
        cat_neg_with_feature = defaultdict(int)

        for report_id, value in report_values:
            cats_for_report = report_to_cats.get(report_id, set())
            for cat in self.categories:
                if cat in cats_for_report:
                    pos_counts[cat][value] += 1
                    cat_pos_with_feature[cat] += 1
                else:
                    neg_counts[cat][value] += 1
                    cat_neg_with_feature[cat] += 1

        # 4. Compute smoothed likelihoods
        self.likelihoods[feature] = {}

        for cat in self.categories:
            self.likelihoods[feature][cat] = {}
            pos_total = cat_pos_with_feature[cat]
            neg_total = cat_neg_with_feature[cat]

            for val in values:
                pc = pos_counts[cat][val]
                nc = neg_counts[cat][val]

                lik_pos = (pc + alpha) / (pos_total + alpha * n_values)
                lik_neg = (nc + alpha) / (neg_total + alpha * n_values)

                self.likelihoods[feature][cat][val] = (lik_pos, lik_neg)

        print(f"Computed likelihoods for {feature}: {n_values} values "
              f"(pos + neg for {len(self.categories)} categories)")

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
                val = row[i + 1]
                if val and fname in self.features:
                    features_provided[fname] = val

            if not features_provided:
                continue

            # Quick posterior computation via _predict_raw
            posteriors = self._predict_raw(features_provided)
            if posteriors:
                top1_probs.append(max(posteriors.values()))

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

    def _unseen_likelihood(self, feature: str, cat: str, label: int) -> float:
        """
        Proper Laplace smoothing for unseen feature values.

        For a value never seen during training, the count is 0.
        We add +1 to n_values to account for the unseen value itself.
        """
        alpha = self.smoothing_alpha
        n_values = len(self.feature_value_sets.get(feature, [])) + 1  # +1 for unseen
        pos_count, neg_count = self.category_report_counts.get(cat, (1, 1))
        denominator = pos_count if label == 1 else neg_count
        return alpha / (denominator + alpha * n_values)

    def _predict_raw(self, features_provided: Dict[str, str]) -> Dict[str, float]:
        """
        Compute binary posteriors for all categories (internal use).

        Returns dict of {category_code: probability}.
        """
        posteriors = {}
        for category in self.categories:
            pos_count, neg_count = self.category_report_counts.get(
                category, (1, 1)
            )
            total = pos_count + neg_count

            log_pos = math.log(pos_count / total + 1e-10)
            log_neg = math.log(neg_count / total + 1e-10)

            for feature, value in features_provided.items():
                if feature in self.likelihoods and category in self.likelihoods[feature]:
                    lik_tuple = self.likelihoods[feature][category].get(value)
                    if lik_tuple is not None:
                        lik_pos, lik_neg = lik_tuple
                    else:
                        # Unseen value — proper Laplace
                        lik_pos = self._unseen_likelihood(feature, category, 1)
                        lik_neg = self._unseen_likelihood(feature, category, 0)

                    log_pos += math.log(lik_pos + 1e-10)
                    log_neg += math.log(lik_neg + 1e-10)

            logit = log_pos - log_neg
            posteriors[category] = _sigmoid(logit)

        return posteriors

    def predict(self, top_k: int = 5, **kwargs) -> List[Dict]:
        """
        Predict accident category probabilities given flight profile.

        Uses Binary Relevance Naive Bayes — each category gets an independent
        probability via sigmoid. Multiple categories CAN have high probability
        simultaneously. Probabilities do NOT sum to 1.

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

        posteriors = self._predict_raw(features_provided)

        # Sort and return top k
        sorted_cats = sorted(posteriors.items(), key=lambda x: -x[1])[:top_k]

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
        Returns the positive-class likelihoods P(val|c=1).
        """
        if feature not in self.likelihoods:
            return {}
        if category not in self.likelihoods[feature]:
            return {}

        dist = self.likelihoods[feature][category]
        # Extract positive likelihood from (lik_pos, lik_neg) tuple
        return {k: round(v[0], 4) for k, v in sorted(dist.items(), key=lambda x: -x[1][0])}

    # ------------------------------------------------------------------
    # Persistence: save/load model to/from SQLite
    # ------------------------------------------------------------------

    def save_to_db(self):
        """
        Save computed priors and likelihoods to SQLite tables.
        Replaces existing data. Saves both pos and neg likelihoods.
        """
        cursor = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()

        # Clear existing
        cursor.execute("DELETE FROM bayes_priors")
        cursor.execute("DELETE FROM bayes_likelihoods")

        # Save priors with positive_count
        for code, prob in self.priors.items():
            pos_count = self.category_report_counts[code][0]
            cursor.execute("""
                INSERT INTO bayes_priors (category_code, prior_probability,
                                          positive_count, sample_count, computed_at)
                VALUES (?, ?, ?, ?, ?)
            """, (code, prob, pos_count, self.training_report_count, now))

        # Save likelihoods — 2 rows per (cat, feat, val): label=1 and label=0
        likelihood_count = 0
        for feature, cat_dict in self.likelihoods.items():
            for category, val_dict in cat_dict.items():
                pos_count = self.category_report_counts[category][0]
                neg_count = self.category_report_counts[category][1]
                for value, (lik_pos, lik_neg) in val_dict.items():
                    cursor.execute("""
                        INSERT INTO bayes_likelihoods
                        (category_code, feature_name, feature_value,
                         label, likelihood, sample_count)
                        VALUES (?, ?, ?, 1, ?, ?)
                    """, (category, feature, value, lik_pos, pos_count))
                    cursor.execute("""
                        INSERT INTO bayes_likelihoods
                        (category_code, feature_name, feature_value,
                         label, likelihood, sample_count)
                        VALUES (?, ?, ?, 0, ?, ?)
                    """, (category, feature, value, lik_neg, neg_count))
                    likelihood_count += 1

        self.conn.commit()
        print(f"Saved model: {len(self.priors)} priors, "
              f"{likelihood_count} likelihood entries (x2 for pos/neg)")

    def load_from_db(self):
        """
        Load priors and likelihoods from SQLite tables (fast path).
        Returns True if loaded successfully, False if tables are empty.
        """
        cursor = self.conn.cursor()

        # Load priors
        rows = cursor.execute("""
            SELECT category_code, prior_probability, positive_count, sample_count
            FROM bayes_priors
        """).fetchall()

        if not rows:
            return False

        self.priors = {}
        self.categories = []
        self.category_report_counts = {}
        for code, prob, pos_count, total in rows:
            self.priors[code] = prob
            self.categories.append(code)
            self.training_report_count = total  # same for all rows
            if pos_count is not None:
                self.category_report_counts[code] = (pos_count, total - pos_count)
            else:
                # Fallback for old schema without positive_count
                count = int(round(prob * total))
                self.category_report_counts[code] = (count, total - count)

        # Load likelihoods grouped by label, rebuild (lik_pos, lik_neg) tuples
        rows = cursor.execute("""
            SELECT category_code, feature_name, feature_value, label, likelihood
            FROM bayes_likelihoods
            ORDER BY category_code, feature_name, feature_value, label
        """).fetchall()

        # First pass: collect by (feat, cat, val)
        raw = {}  # {feat: {cat: {val: {label: lik}}}}
        for code, feature, value, label, likelihood in rows:
            raw.setdefault(feature, {}).setdefault(code, {}).setdefault(value, {})[label] = likelihood

        # Second pass: build tuples
        self.likelihoods = {}
        self.feature_value_sets = {}
        for feature, cat_dict in raw.items():
            self.likelihoods[feature] = {}
            values_set = set()
            for cat, val_dict in cat_dict.items():
                self.likelihoods[feature][cat] = {}
                for val, label_dict in val_dict.items():
                    lik_pos = label_dict.get(1, 0.01)
                    lik_neg = label_dict.get(0, 0.01)
                    self.likelihoods[feature][cat][val] = (lik_pos, lik_neg)
                    values_set.add(val)
            self.feature_value_sets[feature] = sorted(values_set)

        # Rebuild feature list from loaded data
        self.features = sorted(self.likelihoods.keys())

        # Compute risk thresholds from loaded model
        self._compute_risk_thresholds()

        print(f"Loaded model: {len(self.priors)} priors, "
              f"{len(self.likelihoods)} features (binary relevance)")
        return True

    # ------------------------------------------------------------------
    # Validation: proper leave-one-out cross-validation
    # ------------------------------------------------------------------

    def validate(self, verbose=True) -> Dict:
        """
        Proper leave-one-out cross-validation on accident reports.

        For each held-out report:
        1. Adjust counts (subtract held-out report from pos/neg counts)
        2. Recompute smoothed likelihoods from adjusted counts
        3. Compute sigmoid posterior
        4. Check if any true L1 category appears in top-k

        Also computes:
        - Prior-only baseline for comparison
        - Expected Calibration Error (ECE)

        Returns:
            Dict with hit@1, hit@3, hit@5, mean_rank, baseline metrics, ECE
        """
        cursor = self.conn.cursor()

        # Get all accident reports with features AND taxonomy
        reports = cursor.execute("""
            SELECT DISTINCT f.report_id, f.aircraft_category, f.season, f.region,
                   f.weather_category, f.time_of_day
            FROM report_features f
            JOIN report_taxonomy t ON f.report_id = t.report_id
            JOIN report_types rt ON f.report_id = rt.report_id
            WHERE t.level = 'L1'
              AND rt.report_type = 'accident'
              AND (f.aircraft_category IS NOT NULL OR f.season IS NOT NULL
                   OR f.region IS NOT NULL OR f.weather_category IS NOT NULL
                   OR f.time_of_day IS NOT NULL)
        """).fetchall()

        if verbose:
            print(f"Running proper LOO cross-validation on {len(reports)} reports...")

        # Pre-fetch all taxonomy assignments
        all_cats = cursor.execute("""
            SELECT report_id, category_code
            FROM report_taxonomy
            WHERE level = 'L1'
        """).fetchall()
        report_true_cats = defaultdict(set)
        for rid, cat in all_cats:
            report_true_cats[rid].add(cat)

        # Pre-compute raw counts for LOO adjustment
        # For each (feature, category, value): how many positive and negative reports
        # We need to know per-report which feature values they have and which cats
        feature_cols = ['aircraft_category', 'season', 'region',
                        'weather_category', 'time_of_day']

        # Build per-report feature map
        report_features = {}
        for row in reports:
            rid = row[0]
            features_provided = {}
            for i, fname in enumerate(feature_cols):
                val = row[i + 1]
                if val and fname in self.features:
                    features_provided[fname] = val
            report_features[rid] = features_provided

        # Pre-compute count structures for efficient LOO
        # pos_val_counts[feat][cat][val] = count of positive reports
        # neg_val_counts[feat][cat][val] = count of negative reports
        # cat_pos_total[feat][cat] = total positive reports with this feature
        # cat_neg_total[feat][cat] = total negative reports with this feature
        pos_val_counts = {}
        neg_val_counts = {}
        cat_pos_total = {}
        cat_neg_total = {}

        for feat in self.features:
            pos_val_counts[feat] = {cat: defaultdict(int) for cat in self.categories}
            neg_val_counts[feat] = {cat: defaultdict(int) for cat in self.categories}
            cat_pos_total[feat] = defaultdict(int)
            cat_neg_total[feat] = defaultdict(int)

        # Fill counts from training data
        for row in reports:
            rid = row[0]
            cats_for_report = report_true_cats.get(rid, set())
            for i, fname in enumerate(feature_cols):
                val = row[i + 1]
                if val and fname in self.features:
                    for cat in self.categories:
                        if cat in cats_for_report:
                            pos_val_counts[fname][cat][val] += 1
                            cat_pos_total[fname][cat] += 1
                        else:
                            neg_val_counts[fname][cat][val] += 1
                            cat_neg_total[fname][cat] += 1

        alpha = self.smoothing_alpha

        # Baseline: prior-only prediction
        baseline_hits = {1: 0, 3: 0, 5: 0}
        prior_ranking = sorted(self.priors.items(), key=lambda x: -x[1])
        baseline_top_codes = [code for code, _ in prior_ranking]

        # Model metrics
        hits = {1: 0, 3: 0, 5: 0}
        ranks = []
        evaluated = 0

        # ECE calibration tracking
        calibration_pairs = []  # (predicted_prob, actual_label)

        for idx, row in enumerate(reports):
            held_out_id = row[0]
            true_cats = report_true_cats.get(held_out_id, set())
            features_provided = report_features[held_out_id]

            if not true_cats or not features_provided:
                continue

            # --- LOO posterior computation ---
            posteriors = {}
            for cat in self.categories:
                # Adjust prior counts
                pos_c, neg_c = self.category_report_counts[cat]
                has_cat = cat in true_cats
                if has_cat:
                    adj_pos = pos_c - 1
                    adj_neg = neg_c
                else:
                    adj_pos = pos_c
                    adj_neg = neg_c - 1
                adj_total = adj_pos + adj_neg

                if adj_total <= 0:
                    posteriors[cat] = 0.5
                    continue

                log_pos = math.log(adj_pos / adj_total + 1e-10)
                log_neg = math.log(adj_neg / adj_total + 1e-10)

                for feat, val in features_provided.items():
                    if feat not in self.likelihoods:
                        continue
                    n_vals = len(self.feature_value_sets.get(feat, []))
                    if n_vals == 0:
                        continue

                    # Adjust feature counts for LOO
                    pc = pos_val_counts[feat][cat].get(val, 0)
                    nc = neg_val_counts[feat][cat].get(val, 0)
                    pt = cat_pos_total[feat][cat]
                    nt = cat_neg_total[feat][cat]

                    if has_cat:
                        # Remove this report from positive class
                        pc_adj = pc - (1 if val == features_provided.get(feat) else 0)
                        pt_adj = pt - 1
                        nc_adj = nc
                        nt_adj = nt
                    else:
                        # Remove this report from negative class
                        pc_adj = pc
                        pt_adj = pt
                        nc_adj = nc - (1 if val == features_provided.get(feat) else 0)
                        nt_adj = nt - 1

                    # Clamp to non-negative
                    pc_adj = max(0, pc_adj)
                    pt_adj = max(0, pt_adj)
                    nc_adj = max(0, nc_adj)
                    nt_adj = max(0, nt_adj)

                    lik_pos = (pc_adj + alpha) / (pt_adj + alpha * n_vals) if pt_adj + alpha * n_vals > 0 else 1.0 / n_vals
                    lik_neg = (nc_adj + alpha) / (nt_adj + alpha * n_vals) if nt_adj + alpha * n_vals > 0 else 1.0 / n_vals

                    log_pos += math.log(lik_pos + 1e-10)
                    log_neg += math.log(lik_neg + 1e-10)

                posteriors[cat] = _sigmoid(log_pos - log_neg)

            # Sort by posterior descending
            sorted_preds = sorted(posteriors.items(), key=lambda x: -x[1])
            predicted_codes = [code for code, _ in sorted_preds]

            # Model hit check
            for k in [1, 3, 5]:
                top_k_codes = set(predicted_codes[:k])
                if true_cats & top_k_codes:
                    hits[k] += 1

            # Baseline hit check
            for k in [1, 3, 5]:
                if true_cats & set(baseline_top_codes[:k]):
                    baseline_hits[k] += 1

            # Best rank
            best_rank = len(predicted_codes) + 1
            for true_cat in true_cats:
                if true_cat in predicted_codes:
                    rank = predicted_codes.index(true_cat) + 1
                    best_rank = min(best_rank, rank)
            ranks.append(best_rank)

            # ECE: collect (predicted_prob, actual_label) for every category
            for cat, prob in posteriors.items():
                actual = 1 if cat in true_cats else 0
                calibration_pairs.append((prob, actual))

            evaluated += 1
            if verbose and (idx + 1) % 100 == 0:
                print(f"  Evaluated {idx + 1}/{len(reports)}...")

        if evaluated == 0:
            print("No reports could be evaluated")
            return {}

        # Compute baseline ranks
        baseline_ranks = []
        for row in reports:
            rid = row[0]
            true_cats = report_true_cats.get(rid, set())
            if not true_cats or not report_features[rid]:
                continue
            best_rank = len(baseline_top_codes) + 1
            for tc in true_cats:
                if tc in baseline_top_codes:
                    r = baseline_top_codes.index(tc) + 1
                    best_rank = min(best_rank, r)
            baseline_ranks.append(best_rank)

        # Compute ECE (10-bin)
        ece = self._compute_ece(calibration_pairs, n_bins=10)

        results = {
            'total_evaluated': evaluated,
            'hit_at_1': hits[1] / evaluated,
            'hit_at_3': hits[3] / evaluated,
            'hit_at_5': hits[5] / evaluated,
            'mean_rank': sum(ranks) / len(ranks),
            'median_rank': sorted(ranks)[len(ranks) // 2],
            'baseline_hit_at_1': baseline_hits[1] / evaluated,
            'baseline_hit_at_3': baseline_hits[3] / evaluated,
            'baseline_hit_at_5': baseline_hits[5] / evaluated,
            'baseline_mean_rank': sum(baseline_ranks) / len(baseline_ranks) if baseline_ranks else 0,
            'ece': ece,
        }

        if verbose:
            bh1 = results['baseline_hit_at_1']
            bh3 = results['baseline_hit_at_3']
            bh5 = results['baseline_hit_at_5']
            mh1 = results['hit_at_1']
            mh3 = results['hit_at_3']
            mh5 = results['hit_at_5']
            bmr = results['baseline_mean_rank']
            mmr = results['mean_rank']

            print(f"\n{'='*60}")
            print(f"CROSS-VALIDATION RESULTS ({evaluated} reports, proper LOO)")
            print(f"{'='*60}")
            print(f"                    Model     Baseline    Lift")
            print(f"  Hit@1:            {mh1:5.1%}     {bh1:5.1%}       {mh1/bh1:.2f}x" if bh1 > 0 else f"  Hit@1:            {mh1:5.1%}     {bh1:5.1%}       N/A")
            print(f"  Hit@3:            {mh3:5.1%}     {bh3:5.1%}       {mh3/bh3:.2f}x" if bh3 > 0 else f"  Hit@3:            {mh3:5.1%}     {bh3:5.1%}       N/A")
            print(f"  Hit@5:            {mh5:5.1%}     {bh5:5.1%}       {mh5/bh5:.2f}x" if bh5 > 0 else f"  Hit@5:            {mh5:5.1%}     {bh5:5.1%}       N/A")
            print(f"  Mean rank:        {mmr:.1f}       {bmr:.1f}")
            print()
            print(f"CALIBRATION")
            print(f"  ECE:              {ece:.3f}     (0 = perfect)")
            print()
            print(f"Features: {', '.join(self.features)}")
            print(f"{'='*60}")

        return results

    @staticmethod
    def _compute_ece(pairs: List[Tuple[float, int]], n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error.

        Bins all (predicted_probability, actual_label) pairs into n_bins.
        ECE = weighted average of |avg_predicted - avg_actual| per bin.
        """
        if not pairs:
            return 0.0

        bins = [[] for _ in range(n_bins)]
        for prob, actual in pairs:
            bin_idx = min(int(prob * n_bins), n_bins - 1)
            bins[bin_idx].append((prob, actual))

        total = len(pairs)
        ece = 0.0
        for bin_data in bins:
            if not bin_data:
                continue
            avg_pred = sum(p for p, _ in bin_data) / len(bin_data)
            avg_actual = sum(a for _, a in bin_data) / len(bin_data)
            ece += len(bin_data) / total * abs(avg_pred - avg_actual)

        return ece


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
    conn = sqlite3.connect(db_path, check_same_thread=False)
    model = BayesianRiskModel(conn, train=False)
    if not model.load_from_db():
        print("No saved model found, training from scratch...")
        model = BayesianRiskModel(conn)
    return model


if __name__ == "__main__":
    # Demo the model
    print("Building Binary Relevance Bayesian Risk Model...")
    print("=" * 50)

    model = build_model()

    print("\n" + "=" * 50)
    print("BASE RATES (Prior Probabilities)")
    print("=" * 50)
    for cat in model.get_base_rates(10):
        print(f"  {cat['category_code']:10s} {cat['percentage']:>6s}  {cat['category_name']}")

    print("\n" + "=" * 50)
    print("EXAMPLE PREDICTIONS (Binary Relevance)")
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
        top_k=10,
        aircraft_category="turboprop", season="Winter", region="West",
        weather_category="IMC", time_of_day="Night"
    )
    for r in results:
        print(f"  [{r['risk_level']:8s}] {r['percentage']:>6s}  {r['category_code']:10s} - {r['category_name']}")

    # Show that probabilities don't sum to 1
    all_preds = model.predict(top_k=27)
    total_prob = sum(p['probability'] for p in all_preds)
    print(f"\nSum of all 27 category probabilities: {total_prob:.2f} (NOT 1.0)")

    # Save model
    print("\nSaving model to database...")
    model.save_to_db()

    # Validate
    print("\nRunning cross-validation...")
    model.validate()
