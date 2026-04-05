import numpy as np

class TEDA:
    """Detect outliers in a stream or tabular dataset."""
    # ------------------------------
    # Constructor
    #-------------------------------
    def __init__(self, threshold):
        # Initialize state variables
        self.k = 1
        self.variance = 0
        self.mean = 0
        self.threshold = threshold

    # ------------------------------
    # Internal methods
    #------------------------------- 
    def __calcMean(self, x):
        return ((self.k-1)/self.k)*self.mean + (1/self.k)*x
    
    def __calcVariance(self, x):
        distance_squared = np.square(np.linalg.norm(x - self.mean))
        return ((self.k-1)/self.k)*self.variance + distance_squared*(1/(self.k - 1))
                                     
    def __calcEccentricity(self, x):
        #if(self.variance == 0):
        #    self.variance = 0.00001
            
        if(self.variance == 0):
            new_ecc = 0
            return new_ecc
     
        if (isinstance(x, float)):
            return (1 / self.k) +  (((self.mean - x)*(self.mean - x)) / (self.k *  self.variance))    
        else:
            return (1 / self.k) +  (((self.mean - x).T.dot((self.mean - x))) / (self.k *  self.variance))
            
        
    
    # ------------------------------
    # Run methods
    #-------------------------------
    def run_offline(self, df, features):
        """Run the algorithm offline."""
        
        # Add the is_outlier column to the dataframe
        df['is_outlier'] = 0
        
        # Iterate over dataframe rows
        for index, row in df.iterrows():
            # Build the current sample as a NumPy array
            x = np.array(row[features])
            
            # Update model statistics
            if(self.k == 1):
                self.mean = x
                self.variance = 0
            else:
                # Compute the updated mean
                self.mean = self.__calcMean(x)
                # Compute the updated variance
                self.variance = self.__calcVariance(x)
                # Compute eccentricity and normalized eccentricity
                eccentricity = self.__calcEccentricity(x)
                norm_eccentricity = eccentricity/2
                # Define the outlier threshold
                threshold_ = (self.threshold**2 +1)/(2*self.k)
                
                # Check whether the sample is an outlier
                isOutlier = norm_eccentricity > threshold_

                # Mark the sample as an outlier when the threshold is exceeded
                if (isOutlier):
                    df.at[index, 'is_outlier'] = 1

            # Advance the time index
            self.k = self.k + 1
            

    def run(self, x):
        "Run the algorithm online."""

        is_outlier = 0
        
        # Update model statistics
        if(self.k == 1):
            self.mean = x
            self.variance = 0
            #is_outlier = 1
        else:
            # Compute the updated mean
            self.mean = self.__calcMean(x)
            # Compute the updated variance
            self.variance = self.__calcVariance(x)
            # calculate the eccentricity and nomalized eccentricity
            eccentricity = self.__calcEccentricity(x)
            norm_eccentricity = eccentricity/2
            # Define the outlier threshold
            threshold_ = (self.threshold**2 +1)/(2*self.k)
            
            # Check whether the sample is an outlier
            isOutlier = norm_eccentricity > threshold_

            # Mark the sample as an outlier when the threshold is exceeded
            if (isOutlier):
                is_outlier = 1

        # Advance the time index
        self.k = self.k + 1

        return is_outlier