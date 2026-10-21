"""Write examples/data/{train,test}.csv: a churn table split the wrong way.

    python examples/make_leaky_split.py
    unleak data examples/data/train.csv examples/data/test.csv \
        --target churned --group customer_id --time event_date
"""

import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

OUT = pathlib.Path(__file__).parent / "data"


def main(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    customers = 1500
    rows_per_customer = rng.integers(1, 6, customers)
    customer_id = np.repeat(np.arange(customers), rows_per_customer)
    n = len(customer_id)
    churned_by_customer = rng.random(customers) < 0.2
    churned = churned_by_customer[customer_id].astype(int)
    df = pd.DataFrame({
        "customer_id": customer_id,
        "event_date": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 540, n), unit="D"),
        "monthly_spend": rng.gamma(2.0, 30.0, n).round(2),
        "support_tickets": rng.poisson(1.5, n),
        "avg_session_min": rng.lognormal(2.0, 0.6, n).round(3),
        # recorded *after* a customer cancels: a textbook proxy for the label
        "days_since_cancellation": np.where(churned == 1, rng.integers(1, 90, n), 0),
        "churned": churned,
    })
    train, test = train_test_split(df, test_size=0.25, random_state=seed)   # row-wise, not by customer
    test = pd.concat([test, train.sample(37, random_state=seed)])           # rows copied during a merge
    OUT.mkdir(exist_ok=True)
    train.to_csv(OUT / "train.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)
    print(f"wrote {OUT}/train.csv ({len(train)} rows) and test.csv ({len(test)} rows)")


if __name__ == "__main__":
    main()
