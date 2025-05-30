import os
from time import time

import matplotlib.pyplot as plt
import numpy as np

from defs import T, age_vec, dx, nb_dstn, params, root_dir
from Tools.seq_jacobian import fake_news_jacobian
from Tools.SimpleNKAgent import SimpleNKAgent

T_plots = 50
s_plots = np.array([0, 20, 40])

figure_dir = os.path.join(root_dir, "Figures")


def cohort_jac(age_jac):
    A = age_jac.shape[0]
    T = age_jac.shape[1]

    # Dims are: cohort, time, s
    # with cohorts indexed by their age at time 0.
    c_jac = np.zeros((A, A, T))
    c_jac[:, 0, :] = age_jac[:, 0, :]
    for t in range(1, A):
        c_jac[:(-t), t, :] = age_jac[t:, t, :]

    return c_jac


def ss_dstn_plot(agent, filename=None):
    # Get the steady state distribution
    ss_dstn = agent.transitions.find_steady_state_dstn()
    # and the points of policy functions we want to see
    outs = ["c", "income", "a"]
    labels = {"c": "Consumption", "income": "Post-Tax Income", "a": "Savings"}
    out_points = agent.get_outcomes(outs)

    # Get the average outcomes conditional on age
    avgs = {
        k: np.array(
            [np.divide(np.sum(p * d), d.sum()) for p, d in zip(points, ss_dstn)]
        )
        for k, points in out_points.items()
    }

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(6, 2.5))

    # Age distribution
    age_mass = np.array([d.sum() for d in ss_dstn])
    # Shaded area between the age distribution and the x-axis
    ax[0].fill_between(age_vec, age_mass[:-1], alpha=0.2, color="black")
    ax[0].plot(age_vec, age_mass[:-1], color="black")
    ax[0].set_title("Age Distribution")
    ax[0].set_xlabel("Age")
    ax[0].set_ylabel("Mass of Households")

    # Life cycle trajectories
    for k, v in avgs.items():
        ax[1].plot(age_vec, v[:-1], label=labels[k])
    ax[1].set_title("Age Profiles")
    ax[1].set_xlabel("Age")
    ax[1].set_ylabel("Average Outcomes")
    ax[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    # Save as if filename is provided
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight")
    plt.close()


def get_fn_and_jacs(agent, inputs, outputs, verbose=False):
    jacs = {}
    fn_mats = {}
    for shk_param in inputs:
        if verbose:
            print("Calculating jacobians for " + shk_param)
        t0 = time()
        jacs[shk_param], fn_mats[shk_param] = fake_news_jacobian(
            agent=agent,
            shk_param=shk_param,
            outcome_fns=outputs,
            newborn_dstn=nb_dstn,
            horizon=T,
            dx=dx,
            verbose=False,
        )
        if verbose:
            print("Elapsed time: " + str(time() - t0))

    return jacs, fn_mats


def plot_fn_mats(fn_mats, ages, max_t, filename, short_label=False):

    fig, axs = plt.subplots(1, len(ages), figsize=(6, 2.5))
    for i, age in enumerate(ages):
        axs[i].imshow(
            np.abs(fn_mats[age - age_vec[0], :max_t, :max_t]) > 0, cmap="grey"
        )
        if short_label:
            axs[i].set_title(f"a={age - age_vec[0]}")
        else:
            axs[i].set_title(f"Age {age} (a={age - age_vec[0]})")
        axs[i].set_xlabel("s")
        axs[i].set_ylabel("t")
    fig.tight_layout()
    plt.show()
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight")


def plot_jacobians(input, shk_size, output, agent, jacs, ages, filename):
    age_inds = np.array(ages) - age_vec[0]

    # Get the steady state distribution
    ss_dstn = agent.transitions.find_steady_state_dstn()
    # and the age-specific level of outputs, to normalize responses
    out_points = agent.get_outcomes([output])[output]

    # SS levels
    ss_age_levels = np.array([np.sum(p * d) for p, d in zip(out_points, ss_dstn)])

    # Extract the jacobian for the input and output, and get the agg jacobian. Also
    # apply shock size
    jac = jacs[input][output][age_inds, :T_plots, :] * shk_size
    jac = jac[:, :, s_plots]
    agg_jac = jacs[input][output].sum(axis=0)[:T_plots, s_plots] * shk_size

    # Normalize by the steady state level of the output
    jac /= ss_age_levels[age_inds][:, np.newaxis, np.newaxis]
    agg_jac /= ss_age_levels.sum()

    # Find y axis limits
    y_min = min(jac.min(), agg_jac.min())
    y_max = max(jac.max(), agg_jac.max())
    y_range = y_max - y_min
    y_min -= 0.3 * y_range
    y_max += 0.3 * y_range

    fig, axs = plt.subplots(
        1,
        len(ages) + 1,
        figsize=(6, 2.5),
        sharey=True,
    )

    for j, age in enumerate(ages):
        for si, s in enumerate(s_plots):
            axs[j].plot(jac[j, :, si], label=f"s={s}")
            axs[j].set_title(f"Age {age}, $\\mathcal{{J}}\,[{age_inds[j]}]$")
            axs[j].set_xlabel("t")
            axs[j].set_ylim(y_min, y_max)
            axs[j].hlines(0, xmin=0, xmax=T_plots, color="black", lw=0.5, ls="--")

    axs[0].legend(loc="upper right")

    # Plot the aggregate jacobian
    j = len(ages)
    agg_out = ss_age_levels.sum()
    axs[j].plot(agg_jac / agg_out)
    axs[j].set_title(r"Aggregate, $\mathcal{J}$")
    axs[j].set_xlabel("t")
    axs[j].set_ylim(y_min, y_max)
    axs[j].hlines(0, xmin=0, xmax=T_plots, color="black", lw=0.5, ls="--")

    # Formatting
    axs[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.1f}%"))
    axs[0].set_ylabel(f"Relative Response")

    fig.tight_layout()
    plt.show()
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight")


def plot_cohort_resps(input, shk_size, output, agent, coh_jacs, coh_ages, filename):
    
    shk_times = np.array([0, 20, 40])

    age_inds = np.array(coh_ages) - age_vec[0]

    # Get the steady state distribution
    ss_dstn = agent.transitions.find_steady_state_dstn()
    # and the age-specific level of outputs, to normalize responses
    out_points = agent.get_outcomes([output])[output]

    # SS levels
    ss_age_levels = np.array([np.sum(p * d) for p, d in zip(out_points, ss_dstn)])

    # Extract the cohort jacobian for the input and output. Also
    # apply shock size
    coh_jac = coh_jacs[input][output][age_inds, :T_plots, :] * shk_size
    coh_jac = coh_jac[:, :, shk_times]

    # The ammount by which we should normalize shifts with t.
    denom = np.ones((len(coh_ages), T_plots, 1)) * np.nan
    for i, age_ind in enumerate(age_inds):
        for t in range(T_plots):
            if age_ind + t < len(ss_age_levels):
                denom[i, t, 0] = ss_age_levels[age_ind + t]

    # Normalize by the steady state level of the output
    coh_jac /= denom

    # Find y axis limits
    y_min = np.nanmin(coh_jac)
    y_max = np.nanmax(coh_jac)
    y_range = y_max - y_min
    y_min -= 0.3 * y_range
    y_max += 0.3 * y_range

    fig, axs = plt.subplots(
        1,
        len(coh_ages),
        figsize=(6, 2.5),
        sharey=True,
    )

    for j, age in enumerate(coh_ages):
        for si, s in enumerate(shk_times):
            axs[j].plot(coh_jac[j, :, si], label=f"s={s}")
            axs[j].set_title(f"Cohort {age}")
            axs[j].set_xlabel("t")
            axs[j].set_ylim(y_min, y_max)
            axs[j].hlines(0, xmin=0, xmax=T_plots, color="black", lw=0.5, ls="--")

    axs[-1].legend(loc="upper right")

    # Formatting
    axs[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.1f}%"))
    axs[0].set_ylabel(f"Relative Response")

    fig.tight_layout()
    plt.show()
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight")


def main(plots=True):
    # Create and solve agent (without GE calibration)
    agent = SimpleNKAgent(**params)
    agent.update_solution_terminal()
    agent.solve(verbose=False)
    agent.build_transitions(newborn_dstn=nb_dstn)

    # Make the steady state distribution plot
    if plots:
        ss_plot = os.path.join(figure_dir, "ss_dstn_plot.pdf")
        ss_dstn_plot(agent, filename=ss_plot)

    # Get jacobians and fake news matrices
    agg_inputs = ["R", "wage_rate"]
    outs = ["c"]
    jacs, fn_mats = get_fn_and_jacs(agent, agg_inputs, outs, verbose=True)

    # Plot non-zero elements of representative Fn mats
    if plots:
        fn_plot = os.path.join(figure_dir, "fn_nonzero.pdf")
        plot_fn_mats(fn_mats["R"]["c"], ages=[30, 55, 80], max_t=102, filename=fn_plot)

    # Plot jacobians for interest rate
    if plots:
        R_jac_plot = os.path.join(figure_dir, "R_jacobians.pdf")
        plot_jacobians(
            input="R",
            shk_size=0.01,
            output="c",
            agent=agent,
            jacs=jacs,
            ages=[30, 55],
            filename=R_jac_plot,
        )
        W_jac_plot = os.path.join(figure_dir, "W_jacobians.pdf")
        plot_jacobians(
            input="wage_rate",
            shk_size=0.1,
            output="c",
            agent=agent,
            jacs=jacs,
            ages=[55, 80],
            filename=W_jac_plot,
        )

    # %% Cohort responses
    # Get cohort jacs
    cohort_jacs = {
        inp: {out: cohort_jac(jac) for out, jac in jac_dict.items()}
        for inp, jac_dict in jacs.items()
    }
    # Interest rates
    if plots:
        R_coh_plot = os.path.join(figure_dir, "R_coh_resp.pdf")
        plot_cohort_resps(
            input="R",
            shk_size=0.01,
            output="c",
            agent=agent,
            coh_jacs=cohort_jacs,
            coh_ages=[30, 55, 80],
            filename=R_coh_plot,
        )


if __name__ == "__main__":
    main(plots=True)
