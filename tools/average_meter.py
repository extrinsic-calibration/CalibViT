'''
From Z. Zhuang et al.
https://github.com/ICEORY/PMF
'''

class AverageMeter(object):
    '''Computes and stores the average and current value'''

    def __init__(self):
        """
        Initializes the AverageMeter object with default values.
        """
        self.reset()

    def reset(self):
        """
        Resets all metrics (value, average, sum, and count) to their initial states.
        """
        self.val = 0  # Current value
        self.avg = 0  # Running average
        self.sum = 0  # Sum of all values
        self.count = 0  # Number of updates

    def update(self, val: float, n: int = 1):
        """
        Updates the meter with the current value and the count of instances.
        
        Args:
            val (float): Current value to update the meter with.
            n (int, optional): Number of instances for the current value, default is 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count  # Compute the new average


class RunningAvgMeter(object):
    '''Computes and stores the running average and current value
    avg = hist_val * alpha + (1-alpha) * curr_val
    '''

    def __init__(self, alpha: float = 0.95):
        """
        Initializes the RunningAvgMeter object with a given smoothing factor.
        
        Args:
            alpha (float, optional): Smoothing factor for the running average, default is 0.95.
        """
        self.is_init = False  # Indicates whether the running average is initialized
        self.alpha = alpha  # Smoothing factor
        assert 0 <= alpha <= 1, 'alpha should be in the range [0, 1]'
        self.reset()

    def reset(self):
        """
        Resets the running average to an uninitialized state.
        """
        self.is_init = False
        self.avg = 0

    def update(self, val: float):
        """
        Updates the running average with the new value.
        
        Args:
            val (float): Current value to be included in the running average.
        """
        if self.is_init:
            self.avg = self.avg * self.alpha + (1 - self.alpha) * val
        else:
            self.avg = val  # First update
            self.is_init = True


class AverageListMeter(object):
    '''Computes and stores the average and current value of lists'''

    def __init__(self):
        """
        Initializes the AverageListMeter object with default values.
        """
        self.reset()

    def reset(self):
        """
        Resets all metrics (value, average, sum, and count) for lists to their initial states.
        """
        self.val = []  # List of current values
        self.avg = []  # List of averages
        self.sum = []  # List of sums
        self.count = 0  # Number of updates

    def update(self, val: list, n: int = 1):
        """
        Updates the meter with the current list of values and the count of instances.
        
        Args:
            val (list): List of current values to update the meter with.
            n (int, optional): Number of instances for the current value, default is 1.
        """
        self.val = val  # Update current values
        if self.count == 0:
            self.sum = val.copy()  # Initialize sum with the first list of values
        else:
            for i in range(len(val)):
                self.sum[i] += val[i]  # Update sum with new values
        self.count += n  # Update count of instances
        self.avg = [s / self.count for s in self.sum]  # Compute the new average list
