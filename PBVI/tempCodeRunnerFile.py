    def update_belief(self,belief):
        # 入力には[a_{t-1},o_{t-1}]のときの信念b(s):[s]が入る
        # 正規化項 reg
        sum_s1 = torch.einsum("sj,aos->aoj",self.trans_prob,belief)                         # [s,s'] * [a,o,s] -> [a,o,s'] 
        sum_s2 = torch.einsum("sj,jao->sao",self.trans_prob,self.obs_prob)                  # [s,s'] * [s',a,o] -> [s,a,o]
        
        # 正規化項
        reg = torch.einsum("aos,sao->ao",belief,sum_s2)                                     # [a,o,s] * [s,a,o] -> [a,o]    
        
        # 信念の更新
        belief_not_reg = torch.einsum("jao,aoj->aoj",self.obs_prob,sum_s1)                  # [s,a,o] * [a,o,s] -> [a,o,s]  
        update_belief = belief_not_reg / reg.unsqueeze(-1).expand(-1,-1,self.num_states)           # [a,o,s] / ([a,o] -> [a,o,s])
 
        self.belief = update_belief
        
        return update_belief