import torch
from torch import nn
import numpy as np
import config
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


class ClassifierNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ClassifierNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)
        self.net = nn.Sequential(self.layer1, nn.ReLU(), self.layer2, nn.Sigmoid())

    def forward(self, x):
        return self.net(x)

    def hidden(self, x):
        return self.layer1(x)


class MLPClassifier():
    def __init__(self, dataset_l, l_label, dataset_u, dataset_test, test_label, input_dim, logger):
        self.net = ClassifierNetwork(input_dim, config.hidden_dim).to(config.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=config.classifier_learning_rate)

        self.dataset_l = dataset_l
        self.l_label = l_label
        self.dataset_u = dataset_u
        self.dataset_test = dataset_test
        self.test_label = test_label

        self.refresh_num = config.refresh_num

        self.pre_auc_pr = 0
        self.current_data = torch.tensor([])
        self.state = torch.tensor([])
        self.reset()

        self.logger = logger

    def sample(self):
        candidate = np.random.choice([i for i in range(len(self.dataset_u))], size=config.sample_size,
                                     replace=False)
        return self.dataset_u[candidate]

    def reset(self):
        self.current_data = self.sample()
        self.pre_auc_pr = self.calculate_reward()
        self.state = self.net.hidden(self.current_data)
        return self.state

    def step(self, score_change, tot_steps):
        original_score = self.net(self.current_data)
        corrected_score = original_score.detach() + torch.tensor(score_change).to(config.device)
        self.retrain(corrected_score)

        new_auc_pr = self.calculate_reward()
        reward = new_auc_pr - self.pre_auc_pr
        self.pre_auc_pr = new_auc_pr

        self.current_data = self.sample()
        self.state = self.net.hidden(self.current_data)

        test_auc_roc, test_auc_pr = self.test()
        self.logger.log_var("auc_roc", test_auc_roc, tot_steps)
        self.logger.log_var("auc_pr", test_auc_pr, tot_steps)
        return self.state, reward, False

    def test(self):
        anomaly_score = self.net(self.dataset_test)
        auc_roc = roc_auc_score(self.test_label, anomaly_score.cpu().detach())
        precision, recall, _thresholds = precision_recall_curve(self.test_label, anomaly_score.cpu().detach())
        auc_pr = auc(recall, precision)

        return auc_roc, auc_pr

    def calculate_reward(self):
        anomaly_score = self.net(self.dataset_l)
        precision, recall, _thresholds = precision_recall_curve(self.l_label, anomaly_score.cpu().detach())
        reward = auc(recall, precision)

        return reward

    def retrain(self, corrected):
        for ite in range(self.refresh_num):
            original = self.net(self.current_data)
            self.opt.zero_grad()
            loss = nn.CrossEntropyLoss()
            l = loss(original, corrected)
            l.backward()
            self.opt.step()
