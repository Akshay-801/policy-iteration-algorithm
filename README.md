# POLICY ITERATION ALGORITHM

## AIM
To implement the Policy Iteration Algorithm to find the optimal car-moving policy and state-value function for Jack’s Car Rental problem using Dynamic Programming.

## PROBLEM STATEMENT
We are assigned the task of creating a Reinforcement Learning agent to evaluate policies for Jack's Car Rental problem. Jack manages two car rental locations. Every day, some number of customers arrive at each location to rent cars, and some number of rented cars are returned. Renting out a car yields a reward of $10. Overnight, Jack can move cars between the two locations to better meet the demand the next day. Moving a car costs $2.

The environment has a maximum capacity of 20 cars at each location and a maximum of 5 cars can be moved overnight. The requests and returns at each location follow a Poisson distribution. The agent must evaluate different policies dictating how many cars to transfer between locations overnight to maximize the expected discounted return.

## POLICY ITERATION ALGORITHM
1. Initialization: Initialize the value function $V(s) = 0$ and the policy $\pi(s) = 0$ for all states $s \in S$.
2. Policy Evaluation: For the current policy $\pi$, calculate the value of each state $V(s)$. Iteratively update $V(s)$ using the Bellman expectation equation until the change between iterations falls below a small threshold ($\theta$).
3. Policy Improvement: For each state $s$, determine the best action $a$ by calculating the expected return for all possible actions using the updated $V(s)$. Update $\pi(s)$ to the action that yields the highest expected return (acting greedily).
4. Convergence Check: If the policy $\pi$ changes for any state during step 3, return to step 2 with the new policy. If the policy remains stable (no changes), the algorithm has converged, and the current $\pi(s)$ and $V(s)$ are optimal.

## POLICY IMPROVEMENT FUNCTION
### Name : Akshay M
### Register Number: 212224240009
```python
def policy_improvement(env, V, policy, gamma=0.9):
    policy_stable = True
    
    for i in range(env.max_cars + 1):
        for j in range(env.max_cars + 1):
            state = (i, j)
            old_action = policy[i, j]
            
            action_returns = []
            for action in env.actions:
                if (action >= 0 and i >= action) or (action < 0 and j >= abs(action)):
                    expected_ret = env.expected_return(state, action, V, gamma)
                    action_returns.append(expected_ret)
                else:
                    action_returns.append(-float('inf'))
                    
            best_action = env.actions[np.argmax(action_returns)]
            policy[i, j] = best_action
            
            if old_action != best_action:
                policy_stable = False
                
    return policy_stable, policy

```

## POLICY ITERATION FUNCTION
### Name: Akshay M
### Register Number: 212224240009
```python
Include the policy iteration function






```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
</br>
</br>

### 2. Policy, Value function and success rate for the Improved Policy
</br>
</br>

### 3. Policy, Value function and success rate after policy iteration
</br>
</br>


## RESULT:

Write your result here
