import torch
import itertools

# デバイスの設定
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

class Model():
    def __init__(self, sigma, num_people):
        self.sigma = sigma
        self.n = num_people
        
        # 状態、行動、観測空間の生成
        self.attentions = self.generate_weight_list(self.n, 1)
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
        self.actions = self.generate_weight_list(self.n,2)
        self.observations = self.generate_weight_list(self.n,1)
        
        self.num_attentions = self.attentions.shape[0]
        self.num_comforts = self.comforts.shape[0]
        self.num_follows = self.follows.shape[0]
        
        self.num_states = self.states.shape[0]
        self.num_actions = self.actions.shape[0]
        self.num_observations = self.observations.shape[0]
        
        # 状態遷移確率の初期化
        self.trans_prob_attention = torch.eye(self.num_attentions) # dim1 : current attention (縦軸), dim2 : next attention (横軸)
        for i  in range(self.num_attentions):
            self.trans_prob_attention[i] = self.pdf_list(self.attentions[i],self.sigma,self.attentions)  # 状態遷移確率は正規分布でモデル化
        
        # dim1 : current[comfort, follow] (縦軸), dim2 : next[comfort, follow] (横軸) 4*4の行列 [00,01,10,11]
        self.trans_prob_others = torch.tensor([[0.7, 0.1, 0.1, 0.1],  
                                               [0.1, 0.7, 0.1, 0.1],
                                               [0.1, 0.1, 0.7, 0.1],
                                               [0.1, 0.1, 0.1, 0.7]])
                    
        # attentionとothersの遷移確率を結合
        self.trans_prob = torch.einsum("ij,kl->ikjl",self.trans_prob_attention,self.trans_prob_others).reshape(self.num_states,self.num_states)  # [s,s']
        
        # 観測確率の初期化
        self.obs_prob = self.init_obs_prob()  # [s,a,o]
        
        # 信念状態の初期化
        self.belief = (torch.ones(self.num_states) / self.num_states).repeat(self.num_actions,self.num_observations,1)  # [a,o,s]
        
    # 報酬関数の定義
    @staticmethod
    def reward(self,state):
        return state[self.n+1] # 報酬 : comfort
    
    def generate_weight_list(self,n_people,step):
        values = [round(i * 1/step, 2) for i in range(step+1)]  # [0.0, 0.5, 1.0] のリスト
        return torch.tensor([list(comb) for comb in itertools.product(values, repeat=n_people) if round(sum(comb),10)==1.0])
    
    def generate_state_space(self):
        # すべての組み合わせを生成
        state_combinations = list(itertools.product(self.attentions, self.comforts, self.follows))
        
        # テンソルに変換
        state_space = torch.tensor([list(state[0]) + [state[1]] + [state[2]] for state in state_combinations])
        
        return state_space
    
    def pdf_list(self, mean, sigma, weight_list):  # 確率密度関数の生成
        probabilities = []
        for weights in weight_list:
            diff = weights - mean
            prob = (torch.exp(-0.5 * ((diff / sigma) ** 2))/(sigma*torch.sqrt(torch.tensor(2*torch.pi)))).prod().item()
            probabilities.append(prob)
        normalized_probabilities = torch.tensor([prob / sum(probabilities) for prob in probabilities])
        return normalized_probabilities
    
    @staticmethod
    def update_belief(self,belief):
        # 入力には[a_{t-1},o_{t-1}]のときの信念b(s):[s]が入る
        # 正規化項 reg
        sum_s1 = torch.einsum("ij,i->j",self.trans_prob,belief).repeat(self.num_actions,1)     # [s,s'] * [s] -> [s']  -> [a,s'] (整形)
        sum_s2 = torch.einsum("ij,jkl->ikl",self.trans_prob,self.obs_prob)                     # [s,s''] * [s'',a,o] -> [s,a,o]
        reg = torch.einsum("j,jkl->kl",belief,sum_s2)                                          # [s] * [s,a,o] -> [a,o]
        
        # 信念の更新
        belief = torch.einsum("sao,as->aos",self.trans_prob,sum_s1) / reg                      # [s,a,o] / [a,o] -> [a,o,s]       
        
        return belief

    def init_obs_prob(self):
        tensor_shape = (self.num_states, self.num_actions, self.num_observations)
        obs_prob = torch.zeros(tensor_shape)
        
        # 元の順番に戻すためのインデックスを保持, 復元に必要
        follow_0_index = (self.states[:, -1] == 0).nonzero(as_tuple=True)[0]
        follow_1_index = (self.states[:, -1] == 1).nonzero(as_tuple=True)[0]
        
        print(follow_0_index)
        # follow = 1, システムに従う場合
        for i in follow_1_index:
            for j in range(self.num_actions):
                prob = 0.1 * self.pdf_list(self.states[i, :self.n], self.sigma, self.observations) + 0.9 * self.pdf_list(self.actions[j], self.sigma, self.observations)
                obs_prob[i,j,:] = prob
        # follow = 0, システムに従わない場合
        for i in follow_0_index:
            for j in range(self.num_actions):
                prob = 0.9 * self.pdf_list(self.states[i, :self.n], self.sigma, self.observations) + 0.1 * self.pdf_list(self.actions[j], self.sigma, self.observations)
                obs_prob[i,j,:] = prob

        return obs_prob
    
class PBVI():
    def __init__(self, model, horizon, discount_factor=0.9):
        self.model = model
        self.gamma = discount_factor
        
        self.T_dim = horizon
        self.S_dim = model.num_states
        self.A_dim = model.num_actions
        self.O_dim = model.num_observations
        
        self.alpha_vecs = torch.zeros(self.S_dim,self.N_dim)            # 初期のαベクトル  
        self.belief_points = self.generate_belief_points(self.S_dim)    # 信念点のリスト  [s,n]
        self.N_dim = self.belief_points.size(1)                         # 信念点の数
        
    def generate_belief_points(self,num_states, step=0.2):
        """0.2刻みで信念点の組を生成"""
        belief_points = []
        values = [i * step for i in range(int(1 / step) + 1)]
        belief_combinations = itertools.product(values, repeat=num_states)
        for belief in belief_combinations:
            if abs(sum(belief) - 1.0) < 1e-6:  # 合計が1になる組み合わせのみを選択
                belief_points.append(torch.tensor(belief))
        return torch.tensor(belief_points).T
        
    def backup(self):
        best_value = float('-inf')
        best_alpha = None
        best_actions = []
        backup_alpha = torch.zeros(self.S_dim,self.N_dim)
        values = torch.zeros(self.N_dim)
        
        # calc alpha_ao
        alpha_ao = torch.stack([torch.einsum('sj,jao,j->aos',self.model.trans_prob, self.model.obs_prob, self.alpha_vecs[i]) for i in range(self.T)])  # alpha_ao=[a1(s1),a2(s2),...,aT(sT)], a_k(s):[a,o,s],  alpha_ao:[T,a,o,s], j:s'
        # calc alpha_ab
        # r^a(s)
        r_a = torch.zeros(self.A_dim, self.S_dim)
        for i in range(self.S_dim):
            r_a[:,i] = self.model.reward(self.model.states[i])
        
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
            
            best_actions.append(self.model.actions[index])
            backup_alpha[i] = new_alpha
        """
            
        # N_dim に対しても並列化
        b_dot_alpha_ao = torch.einsum('sn,Taos->nTao',self.belief_points,alpha_ao)  # b[:,i]・α^{a,o}=[a1,b1],a_k(s):[a,o,s],  b・α^{a,o}:[T,a,o]'
        argmax_alpha_index = torch.argmax(b_dot_alpha_ao, dim=1)  # argmax_a:[n,a,o]
        
        argmax_alpha = torch.gather(alpha_ao.unsqueeze(0).expand(self.N_dim,-1,-1,-1,-1),
                                dim=0,
                                index=argmax_alpha_index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1,-1,self.A_dim,alpha_ao.size(3),alpha_ao.size(4))).squeeze(1)  # argmax_a:[N,a,o,s]
        sum_alpha_ao = torch.einsum('naos->nas',argmax_alpha)        # sum_alpha_ao:[a,s]
        alpha_ab = r_a.unsqueeze(0).expand(self.N_dim,-1,-1) + self.gamma * sum_alpha_ao  # alpha_ab:[a,s]

        b_dot_alpha_ab = torch.einsum('ns,as->na',self.belief_points,alpha_ab)
        best_action_index = torch.argmax(b_dot_alpha_ab,dim=1)  # best_action_index:[n]
        
        # calc new alpha
        new_alpha = alpha_ab[torch.arange(self.N_dim),best_action_index]
        backup_alpha = new_alpha

        # add best actions
        best_actions = [self.model_actions[i] for i in best_action_index.tolist()]
        
        return backup_alpha, best_actions



        
    def run_backup(self,max_iter=100,epsilon=1e-6):
        for iteration in range(max_iter):
            max_change = 0  # αベクトルの変化量の最大値
            new_backup_alpha = torch.zeros_like(self.alpha_vecs)
            
            for i in range(self.N_dim):
                # b・α^{a,o}
                b_dot_alpha_ao = torch.einsum('s,Taos->Tao', self.belief_points[:, i], self.alpha_ao)  # b[:,i]・α^{a,o}=[a1,b1],a_k(s):[a,o,s,n],  b・α^{a,o}:[T,a,o]'
                argmax_a_index = torch.argmax(b_dot_alpha_ao, dim=0)  # argmax_a:[a,o]

                index = argmax_a_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.alpha_ao.size(2), -1)  # torch.gather のために整形
                argmax_a = torch.gather(self.alpha_ao, 0, index).squeeze(-1)  # argmax_a:[a,o,s]
                sum_alpha_ao = torch.einsum('aos->as', argmax_a)  # sum_alpha_ao:[a,s]
                alpha_ab = self.r_a + self.gamma * sum_alpha_ao  # alpha_ab:[a,s]

                # calc new alpha
                b_dot_alpha_ab = torch.einsum('s,as->a', self.belief_points[:, i], alpha_ab)  # b・alpha_ab:[a]
                index = torch.argmax(b_dot_alpha_ab)
                new_alpha = alpha_ab[index]

                new_backup_alpha[i] = new_alpha

                # 変化量の最大値を更新
                change = torch.norm(new_alpha - self.backup_alpha[i])
                if change > max_change:
                    max_change = change

            self.backup_alpha = new_backup_alpha

            # 収束判定
            if max_change < epsilon:
                break

            
        
            
class POMDP(Model):
    def __init__(self, sigma, num_people):
        super().__init__(sigma, num_people)
        

        
