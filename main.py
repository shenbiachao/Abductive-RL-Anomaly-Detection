import config
import loader
import click
from common import Logger
from common import set_device_and_logger, set_global_seed
from classifier import MLPClassifier
from ddpg_trainer import ReplayBuffer, DDPGTrainer
from ddpg_agent import DDPGAgent
import gym

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--log-dir", default="logs")
@click.option("--gpu", type=int, default=0)
@click.option("--print-log", type=bool, default=True)
@click.option("--seed", type=int, default=4)
@click.option("--info", type=str, default="")
def main(log_dir, gpu, print_log, seed, info):
    # set global seed
    set_global_seed(seed)

    # initialize logger
    env_name = "Anomaly_dectection"
    logger = Logger(log_dir, prefix=env_name + "-" + info, print_to_terminal=print_log)
    logger.log_str("logging to {}".format(logger.log_path))

    # set device and logger
    dev = set_device_and_logger(gpu, logger)
    logger.log_str("Setting device: " + str(dev))

    # load data
    logger.log_str("Loading Data")
    dataset_l, l_label, dataset_u, dataset_test, test_label = loader.load_har()

    # initialize classifier
    classifier = MLPClassifier(dataset_l, l_label, dataset_u, dataset_test, test_label, dataset_l.shape[1], logger)
    # print(classifier.dataset_l.shape, classifier.l_label.shape, classifier.dataset_u.shape)

    # initialize buffer
    logger.log_str("Initializing Buffer")
    buffer = ReplayBuffer(config.sample_size * config.hidden_dim)

    # initialize agent
    logger.log_str("Initializing Agent")
    agent = DDPGAgent(config.sample_size * config.hidden_dim)

    # initialize trainer
    logger.log_str("Initializing Trainer")
    trainer = DDPGTrainer(agent, classifier, buffer, logger)

    logger.log_str("Started training")
    trainer.train()


if __name__ == '__main__':
    main()
