import training.collectData as col 
import training.training as tn

train_actions = ["a", "b"]
new_actions = ["b"]

# col.collectKeypoints(new_actions)
tn.train_neural_net(train_actions)