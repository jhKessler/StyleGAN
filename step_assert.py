

class StepAssert(object):

    def __init__(self, tolerance: int):
        self.tolerance = tolerance
        self.reset()


    def reset(self):
        self.tolerance_left = self.tolerance
        self.biggest_fid_score = None


    def check(self, fid_score: float) -> bool:
        """
        Checks if step should be stopped
        """

        if self.biggest_fid_score is None:
            self.biggest_fid_score = fid_score
            return True

        if fid_score < self.biggest_fid_score:
            self.biggest_fid_score = fid_score
            self.tolerance_left = self.tolerance
            return True
        else:
            self.tolerance_left -= 1
            return self.tolerance_left > 0

    def __call__(self, fid_score: float):
        return self.check(fid_score)


    def get_tolerance(self):
        return self.tolerance_left