import tracemalloc
from time import time

from defs import nb_dstn
from defs import params as lc_params
from Tools.seq_jacobian import fake_news_jacobian
from Tools.SimpleNKAgent import SimpleNKAgent

horizon = 300
dx = 1e-4
input = "R"
output = "c"


def make_ih_params():
    def unlist(par):
        if type(par) == list:
            return par[0]
        else:
            return par

    ih_params = {key: unlist(val) for key, val in lc_params.items()}
    ih_params["cycles"] = 0
    ih_params["T_cycle"] = 1

    ih_params["LivPrb"] = 0.96

    return ih_params


ih_params = make_ih_params()


def LC_benchmark(reps):

    # Create agent
    agent = SimpleNKAgent(**lc_params)
    agent.update_solution_terminal()
    agent.solve(verbose=False)
    agent.build_transitions(newborn_dstn=nb_dstn)

    # Steady state
    t0 = time()
    for _ in range(reps):
        Dss = agent.transitions.find_steady_state_dstn()
    t1 = time()
    steady_state = {"dstn": Dss, "points": agent.get_outcomes([output])}
    results = {
        "num_states": sum([x.size for x in Dss[:-1]]),
        "avg_steady_state_time": (t1 - t0) / reps,
    }

    t0 = time()
    for _ in range(reps):
        fake_news_jacobian(
            agent=agent,
            shk_param=input,
            outcome_fns=[output],
            newborn_dstn=nb_dstn,
            horizon=horizon,
            dx=dx,
            verbose=False,
            steady_state=steady_state,
        )
    t1 = time()

    results.update(
        {
            "avg_jac_time": (t1 - t0) / reps,
            "reps": reps,
        }
    )

    return results


def IH_benchmark(reps):
    agent = SimpleNKAgent(**ih_params)
    agent.update_solution_terminal()
    agent.solve(verbose=False)
    agent.build_transitions(newborn_dstn=nb_dstn)

    # Steady state
    t0 = time()
    for _ in range(reps):
        Dss = agent.transitions.find_steady_state_dstn()
    t1 = time()
    steady_state = {"dstn": Dss[0], "points": agent.get_outcomes([output])}
    results = {
        "num_states": sum([x.size for x in Dss]),
        "avg_steady_state_time": (t1 - t0) / reps,
    }

    t0 = time()
    for _ in range(reps):
        fake_news_jacobian(
            agent=agent,
            shk_param=input,
            outcome_fns=[output],
            newborn_dstn=nb_dstn,
            horizon=horizon,
            dx=dx,
            verbose=False,
            steady_state=steady_state,
        )
    t1 = time()
    results.update(
        {
            "avg_jac_time": (t1 - t0) / reps,
            "reps": reps,
        }
    )
    return results


def main(reps):

    # LC benchmark
    print("LC benchmark")
    lc = LC_benchmark(reps)
    # Run once more to get memory profile
    tracemalloc.reset_peak()
    pre, _ = tracemalloc.get_traced_memory()
    tracemalloc.start()
    _ = LC_benchmark(1)
    post, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    lc["peak_memory_mbs"] = (peak - pre) / (1024 * 1024)
    print(lc)
    print("LC benchmark done")

    # IH benchmark
    print("IH benchmark")
    ih = IH_benchmark(reps)
    # Run once more to get memory profile
    tracemalloc.reset_peak()
    pre, _ = tracemalloc.get_traced_memory()
    tracemalloc.start()
    _ = IH_benchmark(1)
    post, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ih["peak_memory_mbs"] = (peak - pre) / (1024 * 1024)
    print(ih)
    print("IH benchmark done")

    # Print formatted results
    print("Life Cycle")
    print(f"Average steady state time: {lc['avg_steady_state_time']:.4f} seconds")
    print(f"Average jacobian time: {lc['avg_jac_time']:.4f} seconds")
    print(f"Peak memory usage: {lc['peak_memory_mbs']:.4f} MBs")
    print("--------------------------------------------------")
    print("Infinite Horizon")
    print(f"Average steady state time: {ih['avg_steady_state_time']:.4f} seconds")
    print(f"Average jacobian time: {ih['avg_jac_time']:.4f} seconds")
    print(f"Peak memory usage: {ih['peak_memory_mbs']:.4f} MBs")


if __name__ == "__main__":
    main(reps=100)
