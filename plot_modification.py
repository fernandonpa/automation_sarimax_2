def plot_train_fit(title, train_y, train_exog, params, m=_M):
    train_y = train_y.dropna()
    if train_exog is not None:
        train_exog = train_exog.loc[train_y.index]

    p, d, q = int(params.p), int(params.d), int(params.q)
    P, D, Q = int(params.P), int(params.D), int(params.Q)

    result = SARIMAX(
        train_y,
        exog=train_exog,
        order=(p, d, q),
        seasonal_order=(P, D, Q, m),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    fitted = result.get_prediction(
        start=train_y.index[0],
        end=train_y.index[-1],
        exog=train_exog,
    ).predicted_mean
    fitted.index = train_y.index

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(train_y.index, train_y.values, label="Actual", color="#A2CEED")
    ax.plot(fitted.index, fitted.values, label="Fitted", color="orange")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

# -- Best per-horizon params (train fit) ----------------------------
if WEIGHTED_TUNING and not df_tune_h1.empty:
    best_h1 = df_tune_h1.iloc[0]
    plot_train_fit(
        "Horizon 1 - best per-horizon params (train fit)",
        train_y_t1, train_x_t1, best_h1,
    )
else:
    print("Horizon 1 tuning not available - skipping per-horizon fit plot.")

best_h2 = df_tune_h2.iloc[0]
plot_train_fit(
    "Horizon 2 - best per-horizon params (train fit)",
    train_y_t2, train_x_t2, best_h2,
)

# -- Selected params across all horizons (train fit) ----------------
plot_train_fit(
    "Horizon 1 - selected params (train fit)",
    train_y_t1, train_x_t1, best_params,
)
plot_train_fit(
    "Horizon 2 - selected params (train fit)",
    train_y_t2, train_x_t2, best_params,
)
plot_train_fit(
    "Inference - selected params (train fit)",
    train_y_inf, train_x_inf, best_params,
)



##############################################################################

def plot_train_fit(title, train_y, train_exog, params, m=_M):
    train_y = train_y.dropna()
    train_exog = train_exog.loc[train_y.index]

    p, d, q = int(params.p), int(params.d), int(params.q)
    P, D, Q = int(params.P), int(params.D), int(params.Q)

    result = SARIMAX(
        train_y,
        exog=train_exog,
        order=(p, d, q),
        seasonal_order=(P, D, Q, m),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    fitted = result.get_prediction(
        start=train_y.index[0],
        end=train_y.index[-1],
        exog=train_exog,
    ).predicted_mean
    fitted.index = train_y.index

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(train_y.index, train_y.values, label="Actual", color="#A2CEED")
    ax.plot(fitted.index, fitted.values, label="Fitted", color="orange")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

# -- Best per-horizon params (train fit) ----------------------------
if WEIGHTED_TUNING and not df_tune_h1.empty:
    best_h1 = df_tune_h1.iloc[0]
    plot_train_fit(
        "Horizon 1 - best per-horizon params (train fit)",
        train_y_h1, train_exog_h1, best_h1,
    )
else:
    print("Horizon 1 tuning not available - skipping per-horizon fit plot.")

best_h2 = df_tune_h2.iloc[0]
plot_train_fit(
    "Horizon 2 - best per-horizon params (train fit)",
    train_y_h2, train_exog_h2, best_h2,
)

# -- Selected params across all horizons (train fit) ----------------
plot_train_fit(
    "Horizon 1 - selected params (train fit)",
    train_y_h1, train_exog_h1, best_params,
)
plot_train_fit(
    "Horizon 2 - selected params (train fit)",
    train_y_h2, train_exog_h2, best_params,
)
plot_train_fit(
    "Inference - selected params (train fit)",
    train_y_inf, train_exog_inf, best_params,
)
