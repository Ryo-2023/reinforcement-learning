import torch
import itertools
from tqdm import tqdm
import time
import os
from util import util

class Agent():
    def __init__(self, sigma, num_people):
        self.sigma = sigma
        self.n = num_people
        
        # 状態、行動、観測空間の生成
        self.attentions = util.generate_weight_list(self.n, 1)
        self.comforts = torch.tensor([0, 1])
        self.follows = torch.tensor([0, 1])
        
        # すべての組み合わせを生成
        """
        state[0] ~ state[n-1] : attention
        state[n]   : comfort
        state[n+1] : follow
        states: tensor([[0., 0., 1., 0., 0.],
                        [0., 0., 1., 0., 1.],
                        [0., 0., 1., 1., 0.],
                        [0., 0., 1., 1., 1.],
                        [0., 1., 0., 0., 0.],
                        [0., 1., 0., 0., 1.],
                        [0., 1., 0., 1., 0.],
                        [0., 1., 0., 1., 1.],
                        [1., 0., 0., 0., 0.],
                        [1., 0., 0., 0., 1.],
                        [1., 0., 0., 1., 0.],
                        [1., 0., 0., 1., 1.]]) shape:(12, 5)
        """
        self.states = self.generate_state_space()
        self.actions = util.generate_weight_list(self.n,1)
        self.observations = util.generate_weight_list(self.n,1)
        
        self.num_attentions = self.attentions.shape[0]
        self.num_comforts = self.comforts.shape[0]
        self.num_follows = self.follows.shape[0]
        
        self.num_states = self.states.shape[0]
        self.num_actions = self.actions.shape[0]
        self.num_observations = self.observations.shape[0]
        
        # 状態遷移確率の初期化
        self.trans_prob_attention = torch.eye(self.num_attentions) # dim1 : current attention (縦軸), dim2 : next attention (横軸)
        for i  in range(self.num_attentions):
            self.trans_prob_attention[i] = util.pdf_list(self.attentions[i],self.sigma,self.attentions)  # 状態遷移確率は正規分布でモデル化
        
        # dim1 : current[comfort, follow] (縦軸), dim2 : next[comfort, follow] (横軸) 4*4の行列 [00,01,10,11]
        self.trans_prob_others = torch.tensor([[0.6, 0.2, 0.1, 0.1],  
                                               [0.075, 0.6, 0.025, 0.3],
                                               [0.05, 0.05, 0.5, 0.4],
                                               [0.05, 0.05, 0.1, 0.8]])
                    
        # attentionとothersの遷移確率を結合
        self.trans_prob = torch.einsum("ij,kl->ikjl",self.trans_prob_attention,self.trans_prob_others).reshape(self.num_states,self.num_states)  # [s,s']
        
        # 観測確率の初期化
        self.obs_prob = self.init_obs_prob()  # [s,a,o]
        
        # 信念状態の初期化
        self.belief = (torch.ones(self.num_states) / self.num_states).repeat(self.num_actions,self.num_observations,1)  # [a,o,s]
    
    def generate_state_space(self):
        # すべての組み合わせを生成
        state_combinations = list(itertools.product(self.attentions, self.comforts, self.follows))
        
        # テンソルに変換
        state_space = torch.tensor([list(state[0]) + [state[1]] + [state[2]] for state in state_combinations])
        
        return state_space
    
    def update_belief(self,belief):
        # 入力には[a_{t-1},o_{t-1}]のときの信念b(s):[s]が入る
        # 正規化項 reg
        sum_s1 = torch.einsum("sj,aos->aoj",self.trans_prob,belief)                         # [s,s'] * [a,o,s] -> [a,o,s'] 
        sum_s2 = torch.einsum("sj,jao->sao",self.trans_prob,self.obs_prob)                  # [s,s'] * [s',a,o] -> [s,a,o]
        
        # 正規化項
        reg = torch.einsum("aos,sao->ao",belief,sum_s2)                                     # [a,o,s] * [s,a,o] -> [a,o]    
        
        # 信念の更新
        belief_not_reg = torch.einsum("jao,aoj->aoj",self.obs_prob,sum_s1)                  # [s,a,o] * [a,o,s] -> [a,o,s]  
        update_belief = belief_not_reg / reg.unsqueeze(-1).expand(-1,-1,self.num_states)    # [a,o,s] / ([a,o] -> [a,o,s])
        
        self.belief = update_belief
        
        return update_belief

    def init_obs_prob(self):
        tensor_shape = (self.num_states, self.num_actions, self.num_observations)
        obs_prob = torch.zeros(tensor_shape)
        
        # 元の順番に戻すためのインデックスを保持, 復元に必要
        follow_0_index = (self.states[:, -1] == 0).nonzero(as_tuple=True)[0]
        follow_1_index = (self.states[:, -1] == 1).nonzero(as_tuple=True)[0]
        
        #print(follow_0_index)
        # follow = 1, システムに従う場合
        for i in follow_1_index:
            for j in range(self.num_actions):
                prob = 0.1 * util.pdf_list(self.states[i, :self.n], self.sigma, self.observations) + 0.9 * util.pdf_list(self.actions[j], self.sigma, self.observations)
                obs_prob[i,j,:] = prob
        # follow = 0, システムに従わない場合
        for i in follow_0_index:
            for j in range(self.num_actions):
                prob = 0.9 * util.pdf_list(self.states[i, :self.n], self.sigma, self.observations) + 0.1 * util.pdf_list(self.actions[j], self.sigma, self.observations)
                obs_prob[i,j,:] = prob

        return obs_prob
    
class Environment:
    def __init__(self,agent,current_state):
        self.sigma = agent.sigma
        self.agent = agent
        self.current_state = self.init_state()
    
    def init_state(self):
        # 環境の初期状態
        init_state = self.agent.states[0]   # 初期状態として最初の状態を選択
        return init_state
    
    def sample_obs(self,next_state,action):
        if next_state[4] == 1:
            normalized_prob = util.pdf_list(action, self.sigma, self.agent.observations)
            sight_sample_follow = self.agent.observations[torch.multinomial(torch.tensor(normalized_prob), 1).item()]
            return sight_sample_follow
        elif next_state[4] == 0:
            normalized_prob = util.pdf_list(next_state[:3], self.sigma, self.agent.observations)
            sight_sample_not_follow = self.agent.observations[torch.multinomial(torch.tensor(normalized_prob), 1).item()]    # torch.multinomial:与えられた確率分布に基づきサンプリング
            return sight_sample_not_follow
    
    def sample_state(self,state, action):
        # attention
        normalized_prob = util.pdf_list(state[:3], self.sigma, self.agent.attentions)
        next_attention = self.agent.attentions[torch.multinomial(normalized_prob, 1).item()]

        # current index of comfort and follow
        current_index = int(state[3].item() * 2 + state[4].item())
        sample = torch.multinomial(self.agent.trans_prob_others[current_index], 1).item()
        
        # indexからcomfortとfollowを取得
        next_comfort = torch.tensor([sample // 2])
        next_follow = torch.tensor([sample % 2])
        
        # サンプリング結果
        next_state = torch.cat((next_attention, next_comfort, next_follow))

        return next_state
    
    # 報酬関数の定義
    def reward(self,state):
        return state[3] # 報酬 : comfort
    
    def step(self,action):
        next_state = self.sample_state(self.current_state, action)
        
        reward = self.reward(next_state)
        
        obs = self.sample_obs(next_state,action)
        
        # 現状態の更新
        self.current_state = next_state
        
        return next_state, reward, obs
    
class PBVI():
    def __init__(self, Agent, env, horizon, discount_factor=0.95):
        self.Agent = Agent
        self.env = env
        self.gamma = discount_factor
        
        self.T_dim = horizon
        self.S_dim = Agent.num_states
        self.A_dim = Agent.num_actions
        self.O_dim = Agent.num_observations
        
        self.belief_points = self.generate_belief_points()    # 信念点のリスト  [s,n]
        self.N_dim = self.belief_points.size(1)               # 信念点の数
        
        self.alpha_vecs = torch.zeros(self.S_dim,self.N_dim)  # 初期のαベクトル  
        
        self.best_actions = []  # 最適な行動のリスト
        
        # r^a(s)
        self.r_a = torch.zeros(self.A_dim, self.S_dim)
        for i in range(self.S_dim):
            self.r_a[:,i] = self.env.reward(self.Agent.states[i])
        
    def generate_belief_points(self,step=1/4):
        """0.2刻みで信念点の組を生成"""
        values = [round(i * step,2) for i in range(int(1/step)+1)]  
        #belief_points = torch.tensor([list(comb) for comb in itertools.product(values,repeat=num_states) if round(sum(comb),10)==1.0]).T
        
        belief_points = []
        belief_combinations = itertools.product(values, repeat=self.S_dim)
        total_combinations = len(values)**self.S_dim
        print("start generating belief points")
        with tqdm(total=total_combinations, desc="Generating belief points") as p_bar:
            for belief in belief_combinations:
                if abs(sum(belief) - 1.0) < 1e-6:  # 合計が1になる組み合わせのみを選択
                    belief_points.append(list(belief))
                p_bar.update(1)
        
        if len(belief_points) == 0:
            raise ValueError("No valid belief points generated. Check the step size and num_states.")
        
        belief_points_tensor = torch.tensor(belief_points).T.clone().detach()
        print("belief_points:", belief_points_tensor)
        
        return belief_points_tensor
            
    def backup(self):
        best_actions = []
        backup_alpha = torch.zeros(self.S_dim,self.N_dim)
        
        # calc alpha_ao
        alpha_ao = torch.stack([torch.einsum('sj,jao,j->aos',self.Agent.trans_prob, self.Agent.obs_prob, self.alpha_vecs[:,i]) for i in range(self.T_dim)])  # alpha_ao=[a1(s1),a2(s2),...,aT(sT)], a_k(s):[a,o,s],  alpha_ao:[T,a,o,s], j:s'
        # calc alpha_ab

        """
        # N_dim に対してループ
        for i in range(self.N_dim):
            # b・α^{a,o}
            b_dot_alpha_ao = torch.einsum('s,Taos->Tao',self.belief_points[:,i],alpha_ao)  # b[:,i]・α^{a,o}=[a1,b1],a_k(s):[a,o,s],  b・α^{a,o}:[T,a,o]'
            argmax_a_index = torch.argmax(b_dot_alpha_ao, dim=0)  # argmax_a:[a,o]
            
            index = argmax_a_index.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,alpha_ao.size(2),-1)  # torch.gather のために整形
            argmax_a = torch.gather(alpha_ao,0,index).squeeze(-1)  # argmax_a:[a,o,s]
            sum_alpha_ao = torch.einsum('aos->as',argmax_a)        # sum_alpha_ao:[a,s]
            alpha_ab = r_a + self.gamma * sum_alpha_ao             # alpha_ab:[a,s]

            # calc new alpha
            b_dot_alpha_ab = torch.einsum('s,as->a',self.belief_points[:,i],alpha_ab)  # b・alpha_ab:[a]
            index = torch.argmax(b_dot_alpha_ab)
            new_alpha = alpha_ab[index]
            
            best_actions.append(self.Agent.actions[index])
            backup_alpha[i] = new_alpha
        """
            
        # N_dim に対しても並列化
        
        # belief_points: [S, N]
        # alpha_ao: [T, A, O, S]
        # b_dot_alpha_ao: [N, T, A, O]
        b_dot_alpha_ao = torch.einsum('sn,Taos->nTao',self.belief_points,alpha_ao)  # b[:,i]・α^{a,o}=[a1,b1],a_k(s):[a,o,s],  b・α^{a,o}:[T,a,o]'
        argmax_alpha_index = torch.argmax(b_dot_alpha_ao, dim=1)  # argmax_a:[n,a,o]　各nでT次元が一番大きい要素を取得
        
        """
        argmax_alpha = torch.gather(alpha_ao.unsqueeze(0).expand(self.N_dim,-1,-1,-1,-1),  # -> [n,T,a,o,s]
                                dim=1,  # T_dim                             
                                index = argmax_alpha_index
                                        .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).
                                        expand(-1,-1,-1,1,alpha_ao.size(2),alpha_ao.size(3))
                                ).squeeze(1)   # argmax_alpha:[n,a,o,s]   
        """
        
        # gather用に整形
        expanded_alpha = alpha_ao.unsqueeze(0).expand(self.N_dim,-1,-1,-1,-1)  # -> [n,T,a,o,s]

        idx = (argmax_alpha_index
            .unsqueeze(1)       # -> [N_dim, 1, A_dim, O_dim]
            .unsqueeze(-1))     # -> [N_dim, 1, A_dim, O_dim, 1]
        idx = idx.expand(-1, expanded_alpha.size(1), -1, -1, expanded_alpha.size(4)) \
                .permute(0,1,2,3,4)  # [N_dim, T_dim, A_dim, O_dim, S_dim]

        argmax_alpha = torch.gather(expanded_alpha, dim=1, index=idx)           # [N_dim, T_dim, A_dim, O_dim, S_dim]
        
        """
        # T_dim に沿って値が同じかどうか確認
        same_values = torch.allclose(argmax_alpha[:,0],argmax_alpha[:,1])
        print("same_values:",same_values)
        """
        
        argmax_alpha = argmax_alpha.index_select(1,torch.tensor([0]))
        argmax_alpha = argmax_alpha.squeeze(1)   # [N_dim, A_dim, O_dim, S_dim]
        #print("argmax_alpha.shape:",argmax_alpha.shape)
        
        sum_alpha_ao = torch.einsum('naos->nas',argmax_alpha)        # sum_alpha_ao:[n,a,s]

        alpha_ab = self.r_a.unsqueeze(0).expand(self.N_dim,-1,-1) + self.gamma * sum_alpha_ao  # [n,a,s]

        b_dot_alpha_ab = torch.einsum('sn,nas->na',self.belief_points,alpha_ab)
        best_action_index = torch.argmax(b_dot_alpha_ab,dim=1)  # best_action_index:[n]
        
        # calc new alpha
        new_alpha = alpha_ab[torch.arange(self.N_dim),best_action_index].T
        backup_alpha = new_alpha

        # add best actions
        best_actions = [self.Agent.actions[i] for i in best_action_index.tolist()]
        
        return backup_alpha, best_actions
        
    def run_backup(self,max_iter=1000,epsilon=1e-12):
        with tqdm(total = max_iter, desc="PBVI") as p_bar:
            for iteration in tqdm(range(max_iter)):
                max_change = 0  # αベクトルの変化量の最大値
                previous_alpha = self.alpha_vecs.clone()
                #print("previous_alpha:",previous_alpha)
                
                # backup
                backup_alpha, best_actions = self.backup()
                
                #print("backup_alpha:",backup_alpha)
                
                # max_changeの計算
                max_change = torch.norm(backup_alpha - previous_alpha, p=float('inf'))
                #print("max_change:",max_change)
                
                # update backup_alpha, best_actions
                self.alpha_vecs = backup_alpha
                self.best_actions = best_actions

                # 収束判定
                if max_change < epsilon:
                    break
                
                p_bar.update(1)
                
    def get_policy(self):
        return self.best_actions
    
    def get_alpha_vecs(self):
        return self.alpha_vecs
    
    def get_belief_points(self):
        return self.belief_points
            
def save_belief(belief_data,save_list):
    belief_s = torch.einsum("aos->s",belief_data)
    belief_s /= belief_s.sum()
    save_list.append(belief_s.tolist())
    
def main():
    # デバイスの設定
    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    
    # 保存先のパスの指定
    save_dir = "E:/sotsuron/venv_sotsuron/src/PBVI/PBVI_data/logs"
    
    filepath_state = os.path.join(save_dir,"data_state.pkl")
    filepath_belief = os.path.join(save_dir,"data_belief.pkl")
    filepath_action = os.path.join(save_dir,"data_action.pkl")
    filepath_obs = os.path.join(save_dir,"data_obs.pkl")
    
    sigma = 1
    num_people = 3
    horizon = 10
    
    # agentの生成
    agent = Agent(sigma,num_people)
    print("Finish Agent generation")
    
    # environmentの生成
    env = Environment(agent,agent.states[0])
    print("Finish Environment generation")
    
    # PBVIの生成
    pbvi = PBVI(agent,env,horizon)
    pbvi.run_backup()
    
    nsteps = 50   # エージェントと環境のやりとり回数
    
    # データリストの初期化
    state_data  = []
    belief_data = []
    action_data = []
    obs_data    = []
    
    with tqdm(total=nsteps, desc="Simulation steps") as p_bar:
        for i in range(nsteps):
            start_time = time.time()
            
            # PBVIから行動を取得
            best_action = pbvi.get_policy()
            # 適当な行動を選択
            action = agent.actions[0] if len(best_action) == 0 else best_action[i]
            
            # env step : 次状態, 報酬, 観測の取得
            next_state, reward, obs = env.step(action)
            
            # 信念の更新
            belief = agent.update_belief(agent.belief)
            
            # ログの記録
            state_data.append(next_state.clone().tolist())
            save_belief(belief,belief_data)
            action_data.append(action.clone().tolist())
            obs_data.append(obs.clone().tolist())
            
            elapsed = time.time() - start_time
            
            print(f"step:{i+1}, action:{action.tolist()}, next_state:{next_state.tolist()}, reward:{reward}, obs:{obs.tolist()}, elapsed_time:{elapsed}")
            
            p_bar.update(1)
            
    # ログの保存
    os.makedirs(save_dir,exist_ok=True)   # 保存先のディレクトリが存在しない場合は作成
    if filepath_state is not None:
        util.save_data(state_data,filepath_state)
    if filepath_belief is not None:
        util.save_data(belief_data,filepath_belief)
    if filepath_action is not None:
        util.save_data(action_data,filepath_action)
    if filepath_obs is not None:
        util.save_data(obs_data,filepath_obs)

    
if __name__ == "__main__":
    main()
        

        
