'''
From Z. Zhuang et al.
https://github.com/ICEORY/PMF
'''

from .average_meter import RunningAvgMeter

class RemainTime(object):
    '''Estimates the remaining time for training or evaluation based on the average time per iteration'''

    def __init__(self, n_epochs: int):
        """
        Initializes the RemainTime object with the total number of epochs.
        
        Args:
            n_epochs (int): Total number of epochs for the training or evaluation.
        """
        self.n_epochs = n_epochs  # Total number of epochs
        self.timer_avg = {}  # Dictionary to store average time meters for each mode (e.g., 'Train', 'Eval')
        self.total_iter = {}  # Dictionary to store total iterations for each mode

    def update(self, cost_time: float, batch_size: int = 1, mode: str = 'Train'):
        """
        Updates the running average of the time per iteration for the given mode.
        
        Args:
            cost_time (float): Time taken for the current iteration.
            batch_size (int, optional): Batch size for the current iteration (default is 1).
            mode (str, optional): The current mode, either 'Train' or 'Eval' (default is 'Train').
        """
        if mode not in self.timer_avg:
            self.timer_avg[mode] = RunningAvgMeter()  # Initialize RunningAvgMeter for the mode
            self.total_iter[mode] = 0  # Initialize total iteration counter for the mode
        self.timer_avg[mode].update(cost_time)  # Update the running average for the mode

    def reset(self):
        """Resets the timers and iteration counts."""
        self.timer_avg = {}  # Clear the timer averages
        self.total_iter = {}  # Clear the total iteration counts

    def getRemainTime(self, epoch: int, iters: int, total_iter: int, mode: str = 'Train') -> float:
        """
        Calculates the remaining time for training or evaluation based on the average time per iteration.
        
        Args:
            epoch (int): The current epoch number.
            iters (int): The current iteration number within the epoch.
            total_iter (int): The total number of iterations in each epoch.
            mode (str, optional): The current mode, either 'Train' or 'Eval' (default is 'Train').
        
        Returns:
            float: The estimated remaining time in seconds.
        """
        if self.total_iter[mode] == 0:
            self.total_iter[mode] = total_iter  # Initialize total iteration count if it's the first update

        remain_time = 0  # Initialize remaining time
        mode_idx = list(self.timer_avg.keys()).index(mode)  # Get the index of the current mode
        count = 0
        
        for k, v in self.timer_avg.items():
            if k == mode:
                remain_iter = (self.n_epochs - epoch) * self.total_iter[k] - iters
            else:
                if count < mode_idx:
                    remain_iter = (self.n_epochs - epoch - 1) * self.total_iter[k]
                else:
                    remain_iter = (self.n_epochs - epoch) * self.total_iter[k]
            count += 1
            remain_time += v.avg * remain_iter  # Add the estimated remaining time for the current mode
        return remain_time