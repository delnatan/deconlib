import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import mlx.core as mx
    import matplotlib.pyplot as plt
    from toy import shaw, add_poisson_noise
    from deconlib.deconvolution import MatrixOperator, solve_pdhg_with_operator
    return (
        MatrixOperator,
        add_poisson_noise,
        mx,
        plt,
        shaw,
        solve_pdhg_with_operator,
    )


@app.cell
def _(add_poisson_noise, shaw):
    A, x_true, b_exact = shaw(n=64)
    b_noisy = add_poisson_noise(b_exact, peak_photons=1000)

    return A, b_exact, b_noisy, x_true


@app.cell
def _(b_exact, b_noisy, plt, x_recon, x_true):
    plt.plot(b_noisy, 'k-')
    plt.plot(b_exact, 'r--')
    plt.plot(x_true, 'r--')
    plt.plot(x_recon, 'm-')
    return


@app.cell
def _(A, MatrixOperator, b_noisy, mx, solve_pdhg_with_operator):
    op = MatrixOperator(A)
    result = solve_pdhg_with_operator(
        mx.array(b_noisy),
        op,
        alpha=0.01,
        num_iter=300,
        regularization="identity",
        loss_type="gaussian",
        tol=1e-3,
        norm="L1",
    )

    x_recon = result.restored
    return (x_recon,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
