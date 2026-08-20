import numpy as np
import pandas as pd

np.random.seed(42)

N = 25000

df = pd.DataFrame()

# -----------------------------
# Customer Information
# -----------------------------
df["CustomerID"] = range(100001,100001+N)

df["Age"] = np.random.randint(21,66,N)

df["Gender"] = np.random.choice(
    ["Male","Female"],
    N
)

df["MaritalStatus"] = np.random.choice(
    ["Single","Married","Divorced"],
    N,
    p=[0.35,0.55,0.10]
)

df["Education"] = np.random.choice(
    ["HighSchool","Graduate","PostGraduate","PhD"],
    N,
    p=[0.20,0.45,0.30,0.05]
)

df["Occupation"] = np.random.choice(
    ["Salaried","Business","SelfEmployed","Student","Retired"],
    N,
    p=[0.45,0.20,0.20,0.10,0.05]
)

df["EmploymentYears"] = np.random.randint(0,41,N)

# -----------------------------
# Financial
# -----------------------------

df["AnnualIncome"] = np.random.randint(
    180000,
    8000000,
    N
)

df["MonthlyExpenses"] = np.random.randint(
    10000,
    250000,
    N
)

df["Savings"] = np.random.randint(
    0,
    10000000,
    N
)

df["Investments"] = np.random.randint(
    0,
    5000000,
    N
)

df["ExistingLoans"] = np.random.randint(
    0,
    6,
    N
)

df["ExistingEMI"] = np.random.randint(
    0,
    120000,
    N
)

df["LoanAmount"] = np.random.randint(
    50000,
    5000000,
    N
)

df["LoanPurpose"] = np.random.choice(
    [
        "Home",
        "Car",
        "Business",
        "Education",
        "Medical",
        "Personal"
    ],
    N
)

df["LoanTenure"] = np.random.choice(
    [12,24,36,48,60,84,120,180,240],
    N
)

# -----------------------------
# Credit
# -----------------------------

df["CreditScore"] = np.random.randint(
    300,
    901,
    N
)

df["CreditCardLimit"] = np.random.randint(
    50000,
    1500000,
    N
)

df["CreditCardUsage"] = np.random.randint(
    0,
    150000,
    N
)

df["CreditUtilization"] = np.random.randint(
    0,
    101,
    N
)

df["TransactionsPerMonth"] = np.random.randint(
    5,
    300,
    N
)

df["OnlineTransactions"] = np.random.randint(
    0,
    200,
    N
)

df["ATMWithdrawals"] = np.random.randint(
    0,
    25,
    N
)

df["LatePayments"] = np.random.randint(
    0,
    12,
    N
)

df["MissedEMIs"] = np.random.randint(
    0,
    6,
    N
)

df["BankBalance"] = np.random.randint(
    0,
    2000000,
    N
)

df["PropertyValue"] = np.random.randint(
    0,
    50000000,
    N
)

df["Dependents"] = np.random.randint(
    0,
    6,
    N
)

df["Region"] = np.random.choice(
    ["North","South","East","West"],
    N
)

df["CityTier"] = np.random.choice(
    ["Tier1","Tier2","Tier3"],
    N
)

df["PreviousDefaults"] = np.random.randint(
    0,
    4,
    N
)

# -----------------------------
# Intelligent Target Generation
# -----------------------------

risk = (
    (df["CreditScore"] < 600).astype(int)
    + (df["LatePayments"] > 5).astype(int)
    + (df["MissedEMIs"] > 2).astype(int)
    + (df["ExistingLoans"] > 3).astype(int)
    + (df["AnnualIncome"] < 500000).astype(int)
)

df["FraudFlag"] = np.where(
    risk >= 4,
    "Yes",
    "No"
)

segment = []

for r in risk:
    if r <= 1:
        segment.append("Premium")
    elif r <=3:
        segment.append("Regular")
    else:
        segment.append("Risky")

df["CustomerSegment"] = segment

approval = []

for r in risk:
    if r <=2:
        approval.append("Yes")
    else:
        approval.append("No")

df["LoanApproved"] = approval

df.to_csv(
    "data/raw/loan_risk_dataset.csv",
    index=False
)

# print(df.head())

# print(df.shape)

print(df.columns.tolist())