# Kfcc -- fcc Kalman filter for Near Real Time Burnt Area

Kfcc is an algorithm for producing a probabilsitic UQ BA product which is near real time. The algorithm is sensor agnostic


## Ideas
-- Might need to include atmos corr -- eg like Kafka
-- Priors based on active fires with Beta model
-- Maybe additional prior on landcovers... need external dataset for validity? or an estimate of LC unc?
-- Use a logistic regression type thing to convert fcc, a0, a1 to Pb with marginalisation of uncertainties and prior to generate the
predictive posterior estimate of Pb



## Prototype detection alg

1. First derive a good initial condition from a some Kalman smoother over last $N$ days.
2. Use this to make a prediction for the new day $t$. We use the Kalman update with a process model derived from the initial conditions.
3. If an observation:
    1.  Compare the prediction the observation. If matches fcc increase Q so $x_t$ matches observation.
    2.  Also maybe do some spatial thing on this
4. If no observation:
    1. Do normal kalman gain update
5. No run the kalman filter backwards to update the initial condition removing the first day.