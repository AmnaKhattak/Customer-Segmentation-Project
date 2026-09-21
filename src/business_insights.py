import pandas as pd


def create_cluster_summary(
    df,
    features,
    labels
):

    result = df.copy()

    result["Cluster"] = labels

    summary = result.groupby("Cluster")[features].mean()

    summary["Customer Count"] = result.groupby(
        "Cluster"
    ).size()

    return result, summary


def generate_business_insights(
    summary,
    features
):

    age_col = features[0]
    income_col = features[1]
    spending_col = features[2]

    insights = []

    avg_income = summary[income_col].mean()
    avg_spending = summary[spending_col].mean()

    for cluster, row in summary.iterrows():

        income = row[income_col]
        spending = row[spending_col]
        count = int(row["Customer Count"])

        if income >= avg_income and spending >= avg_spending:

            category = "High Income + High Spending"

            recommendation = (
                "Target with premium products, "
                "exclusive offers and loyalty programs."
            )

        elif income >= avg_income and spending < avg_spending:

            category = "High Income + Low Spending"

            recommendation = (
                "Use personalized promotions, "
                "discounts and product recommendations "
                "to increase spending."
            )

        elif income < avg_income and spending >= avg_spending:

            category = "Low Income + High Spending"

            recommendation = (
                "Offer affordable products, "
                "bundles and frequent promotional campaigns."
            )

        elif income < avg_income and spending < avg_spending:

            category = "Low Income + Low Spending"

            recommendation = (
                "Use low-cost marketing strategies "
                "and basic promotional offers."
            )

        else:

            category = "Average Customers"

            recommendation = (
                "Maintain regular engagement "
                "and personalized communication."
            )

        insights.append({
            "Cluster": cluster,
            "Segment": category,
            "Customer Count": count,
            "Average Age": round(row[age_col], 2),
            "Average Income": round(income, 2),
            "Average Spending Score": round(spending, 2),
            "Recommendation": recommendation
        })

    return pd.DataFrame(insights)
