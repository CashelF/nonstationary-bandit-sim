import sys
from matplotlib import pyplot as plt

filename = sys.argv[1]

with open(filename,'r') as f:
    lines = f.readlines()

    sample_average = {
        'average_rs': [float(n) for n in lines[0].strip().split()],
        'average_best_action_taken': [float(n) for n in lines[1].strip().split()],
    }
    constant = {
        'average_rs': [float(n) for n in lines[2].strip().split()],
        'average_best_action_taken': [float(n) for n in lines[3].strip().split()],
    }

    assert len(sample_average['average_rs']) == len(sample_average['average_best_action_taken']) == \
        len(constant['average_rs']) == len(constant['average_best_action_taken']) == 10000

    fig,axes = plt.subplots(2,1)

    # Plotting the average rewards
    axes[0].plot(sample_average['average_rs'], color='blue', label='Sample Average')
    axes[0].plot(constant['average_rs'], color='orange', label='Constant Step-Size')

    # Plotting the average best action taken
    axes[1].plot(sample_average['average_best_action_taken'], color='blue', label='Sample Average')
    axes[1].plot(constant['average_best_action_taken'], color='orange', label='Constant Step-Size')

    # Adding legends to clarify which line is which
    axes[0].legend()
    axes[1].legend()

    # Set titles for clarity
    axes[0].set_title('Average Rewards')
    axes[1].set_title('Average Optimal Action Taken')
    axes[1].set_ylim([0., 1.])

    fig.show()
    _ = input()