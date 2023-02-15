import matplotlib.pyplot as plt

def make_plots(logdir):
    logdir = str(logdir)[2:-2]
    input_file = logdir + "/progress.txt"
    data = []
    with open(input_file) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            data.append(dict(zip(header, map(float, line.strip().split("\t")))))

    x = [d["Epoch"] for d in data]
    y1 = [d["AverageEpRet"] for d in data]
    y2 = [d["AverageEpCost"] for d in data]

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(x, y1, color='b', markersize=4, markeredgewidth=0.5, label='AverageEpRet')
    axs[0].set_xlabel('Epoch', fontsize=12)
    axs[0].set_ylabel('Average Episode Reward', fontsize=12)
    axs[0].grid(True, linestyle='--', color='gray', alpha=0.5)
    axs[1].plot(x, y2, color='r', markersize=4, markeredgewidth=0.5, label='AverageEpCost')
    axs[1].set_xlabel('Epoch', fontsize=12)
    axs[1].set_ylabel('Average Episode Cost', fontsize=12)
    axs[1].grid(True, linestyle='--', color='gray', alpha=0.5)
    fig.suptitle(logdir, fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    args = parser.parse_args()
    make_plots(args.logdir)

if __name__ == "__main__":
    main()