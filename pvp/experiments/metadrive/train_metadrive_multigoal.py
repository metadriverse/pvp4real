import argparse
import os
import os.path as osp

# from pvp.train_metadrive.human_in_the_loop_env import HumanInTheLoopEnv
from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.monitor import Monitor
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.td3.td3 import TD3, ReplayBuffer
from pvp.sb3.td3.policies import TD3Policy
from pvp.sb3.common.vec_env.subproc_vec_env import SubprocVecEnv
from pvp.sb3.common.vec_env.dummy_vec_env import DummyVecEnv
# from drivingforce.human_in_the_loop.common import baseline_eval_config
from pvp.sb3.common.noise import NormalActionNoise
from pvp.utils.utils import get_time_str

from metadrive.envs.multigoal_intersection import MultiGoalIntersectionEnv
from metadrive.envs.gym_wrapper import create_gym_wrapper
import numpy as np


class MultiGoalWrapped(MultiGoalIntersectionEnv):
    current_goal = None

    def step(self, actions):
        o, r, tm, tc, i = super().step(actions)

        o = i['obs/goals/{}'.format(self.current_goal)]
        r = i['reward/goals/{}'.format(self.current_goal)]
        i['route_completion'] = i['route_completion/goals/{}'.format(self.current_goal)]
        i['arrive_dest'] = i['arrive_dest/goals/{}'.format(self.current_goal)]
        i['reward/goals/default'] = i['reward/goals/{}'.format(self.current_goal)]
        i['route_completion/goals/default'] = i['route_completion/goals/{}'.format(self.current_goal)]
        i['arrive_dest/goals/default'] = i['arrive_dest/goals/{}'.format(self.current_goal)]

        return o, r, tm, tc, i

    def reset(self, *args, **kwargs):
        o, i = super().reset(*args, **kwargs)

        # Sample a goal from the goal set
        p = {
            "right_turn": 0.3,
            "left_turn": 0.3,
            "go_straight": 0.1,
            "u_turn": 0.3
        }
        self.current_goal = np.random.choice(list(p.keys()), p=list(p.values()))

        o = i['obs/goals/{}'.format(self.current_goal)]
        i['route_completion'] = i['route_completion/goals/{}'.format(self.current_goal)]
        i['arrive_dest'] = i['arrive_dest/goals/{}'.format(self.current_goal)]
        i['reward/goals/default'] = i['reward/goals/{}'.format(self.current_goal)]
        i['route_completion/goals/default'] = i['route_completion/goals/{}'.format(self.current_goal)]
        i['arrive_dest/goals/default'] = i['arrive_dest/goals/{}'.format(self.current_goal)]

        return o, i



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="TEST", type=str, help="The experiment name.")
    parser.add_argument("--ckpt", default="", type=str)
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    # parser.add_argument("--penalty", default=2.0, type=float)
    parser.add_argument("--lr", default=1e-4, type=float)
    # parser.add_argument("--driving_reward", default=1.0, type=float)
    args = parser.parse_args()

    # ===== Setup some meta information =====
    exp_name = args.exp_name
    seed = int(args.seed)
    use_wandb = args.wandb
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    project_name = "brandon"
    team_name = "drivingforce"

    experiment_batch_name = exp_name
    trial_name = "{}_seed{}_{}".format(exp_name, seed, get_time_str())
    log_dir = osp.join("runs", exp_name, trial_name)
    os.makedirs(osp.join("runs", exp_name), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print("We start logging training data into {}".format(log_dir))

    # ===== Setup the config =====
    config = dict(
        # Environment config
        # env_config={"main_exp": False, "horizon": 1500},

        # Algorithm config
        algo=dict(
            policy=TD3Policy,
            replay_buffer_class=ReplayBuffer,  ###
            replay_buffer_kwargs=dict(),
            policy_kwargs=dict(net_arch=[400, 300]),
            env=None,
            learning_rate=args.lr,
            optimize_memory_usage=True,

            learning_starts=10000 if not args.eval else 0,  ###
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            action_noise=NormalActionNoise(mean=np.zeros([2,]), sigma=0.15 * np.ones([2,])),
            # target_policy_noise=0,
            # policy_delay=1,

            tensorboard_log=log_dir,
            create_eval_env=False,

            verbose=2,
            seed=seed,
            device="auto",
        ),

        # Meta data
        project_name=project_name,
        team_name=team_name,
        exp_name=exp_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=log_dir
    )

    # ===== Setup the training environment =====

    def make_train_env(render=False):

        env_config = dict(
            use_render=render,
            manual_control=False,
            vehicle_config=dict(
                show_navi_mark = True,
                show_line_to_navi_mark = True,
                show_lidar = False,
                show_side_detector = True,
                show_lane_line_detector = True,
            ),
            debug=render,
            # accident_prob=0.0,
            # traffic_density=0.1,
            decision_repeat=5,
            horizon=500,  # to speed up training
            #
            # out_of_road_penalty=0.5,
            # out_of_route_penalty=0.5,
            #
            # map_config=dict(lane_num=1),
        )

        return create_gym_wrapper(MultiGoalWrapped)(env_config)


    train_env = make_train_env(render=args.eval)
    train_env = Monitor(env=train_env, filename=log_dir)
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # eval_env = DummyVecEnv([make_eval_env])
    eval_env = None

    # ===== Setup the callbacks =====
    callbacks = [
        CheckpointCallback(
            name_prefix="rl_model",
            verbose=1,
            save_freq=10000,
            save_path=osp.join(log_dir, "models")
        )
    ]
    if use_wandb:
        callbacks.append(
            WandbCallback(
                trial_name=trial_name,
                exp_name=experiment_batch_name,
                team_name=team_name,
                project_name=project_name,
                config=config
            )
        )

    callbacks = CallbackList(callbacks)

    if args.eval:
        # eval_env = SubprocVecEnv([])
        # eval_env = SubprocVecEnv([lambda: _make_eval_env(False)])
        config["algo"]["learning_rate"] = 0.0
        config["algo"]["train_freq"] = (1, "step")

    # ===== Setup the training algorithm =====
    if args.ckpt:
        model = TD3.load(args.ckpt, **config["algo"])
    else:
        model = TD3(**config["algo"])

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=300_0000,
        callback=callbacks,
        reset_num_timesteps=True,

        # eval
        eval_env=eval_env,
        # eval_freq=5000,
        eval_freq=1,
        n_eval_episodes=30,
        eval_log_path=log_dir,

        # logging
        tb_log_name=exp_name,  # Should place the algorithm name here!
        log_interval=4,
    )
