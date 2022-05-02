import numpy as np
import padasip as pa

# class FilterNLMS(pa.filters.base_filter.AdaptiveFilter):
# This is a function they removed from the main branch of the library
class AdaptiveFilter(pa.filters.base_filter.AdaptiveFilter):
    def explore_learning(self, d, x, mu_start=0, mu_end=1., steps=100, ntrain=0.5, epochs=1, criteria="MSE", target_w=False):
        mu_range = np.linspace(mu_start, mu_end, steps)
        errors = np.zeros(len(mu_range))
        for i, mu in enumerate(mu_range):
            # init
            self.init_weights("zeros")
            self.mu = mu
            # run
            y, e, w = self.pretrained_run(d, x, ntrain=ntrain, epochs=epochs)
            if type(target_w) != bool:
                errors[i] = self.get_mean_error(w[-1]-target_w, function=criteria)
            else:
                errors[i] = self.get_mean_error(e, function=criteria)
        return errors, mu_range    
    

    def get_valid_error(self, x1, x2=-1):
        if type(x2) == int and x2 == -1:
            try:    
                e = np.array(x1)
            except:
                raise ValueError('Impossible to convert series to a numpy array')        
        # two series
        else:
            try:
                x1 = np.array(x1)
                x2 = np.array(x2)
            except:
                raise ValueError('Impossible to convert one of series to a numpy array')
            if not len(x1) == len(x2):
                raise ValueError('The length of both series must agree.')
            e = x1 - x2
        return e

    def logSE(self, x1, x2=-1):
        e = self.get_valid_error(x1, x2)
        return 10*np.log10(e**2)


    def MAE(self, x1, x2=-1):
        e = self.get_valid_error(x1, x2)
        return np.sum(np.abs(e)) / float(len(e))

    def MSE(self, x1, x2=-1):
        e = self.get_valid_error(x1, x2)
        return np.dot(e, e) / float(len(e))

    def RMSE(self, x1, x2=-1):
        e = self.get_valid_error(x1, x2)
        return np.sqrt(np.dot(e, e) / float(len(e)))

    def get_mean_error(self, x1, x2=-1, function="MSE"):
        if function == "MSE":
            return self.MSE(x1, x2)
        elif function == "MAE":
            return self.MAE(x1, x2)
        elif function == "RMSE":
            return self.RMSE(x1, x2)
        else:
            raise ValueError('The provided error function is not known')