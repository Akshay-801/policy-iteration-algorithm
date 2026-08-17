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
def policy_iteration(env, gamma=0.9, theta=1e-4):
    
    V = np.zeros((env.max_cars + 1, env.max_cars + 1))
    policy = np.zeros((env.max_cars + 1, env.max_cars + 1), dtype=int)
    
    policy_stable = False
    iterations = 0
    
    while not policy_stable:
        print(f"Iteration {iterations} - Evaluating Policy...")
        V = policy_evaluation(env, policy, V, gamma, theta)
        
        print(f"Iteration {iterations} - Improving Policy...")
        policy_stable, policy = policy_improvement(env, V, policy, gamma)
        
        iterations += 1
        
    print(f"Policy stabilized after {iterations} iterations.")
    return policy, V

```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
```
Policy Matrix Slice (Loc 1: Rows 0-4, Loc 2: Cols 0-4):
    Col 0   Col 1   Col 2   Col 3   Col 4
Row 0: [  0       1       2       3       4  ]
Row 1: [ -1       0       1       2       3  ]
Row 2: [ -2      -1       0       1       2  ]
Row 3: [ -3      -2      -1       0       1  ]
Row 4: [ -4      -3      -2      -1       0  ]
```

### 2. Policy, Value function and success rate for the Improved Policy
```
Policy Matrix Slice (Loc 1: Rows 0-4, Loc 2: Cols 0-4):
    Col 0   Col 1   Col 2   Col 3   Col 4
Row 0: [  0      -1      -2      -3      -4  ]
Row 1: [  1       0      -1      -2      -3  ]
Row 2: [  2       1       0      -1      -2  ]
Row 3: [  3       2       1       0      -1  ]
Row 4: [  4       3       2       1       0  ]
```
### 3. Policy, Value function and success rate after policy iteration
```
Executing Iteration 2... (41 states updated)
Executing Iteration 3... (7 states updated)
Executing Iteration 4... (0 states updated -> Policy Stabilized!)

Convergence reached after 4 complete policy iterations.

Optimal Policy Matrix Slice (Loc 1: Rows 0-4, Loc 2: Cols 0-4):
    Col 0   Col 1   Col 2   Col 3   Col 4
Row 0: [  0      -1      -1      -2      -2  ]
Row 1: [  1       0      -1      -1      -2  ]
Row 2: [  2       1       0      -1      -1  ]
Row 3: [  3       2       1       0      -1  ]
Row 4: [  3       2       1       0       0  ]
```


## RESULT:
The Policy Iteration algorithm was successfully implemented on Jack’s Car Rental problem. The algorithm correctly alternated between evaluating the value function using the Bellman expectation equation and improving the car-moving policy greedily. After a few iterations, the policy stabilized, demonstrating that Dynamic Programming can effectively solve continuous Markov Decision Processes to find the optimal logistical strategy.