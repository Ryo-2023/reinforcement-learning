import torch
import itertools
import pickle

class util:
    @staticmethod
    def generate_weight_list(n_people,step):
        values = [round(i * 1/step, 2) for i in range(step+1)]  # [0.0, 0.5, 1.0] のリスト
        return torch.tensor([list(comb) for comb in itertools.product(values, repeat=n_people) if round(sum(comb),10)==1.0])
    
    @staticmethod
    def pdf_list(mean, sigma, weight_list):  # 確率密度関数の生成
        probabilities = []
        for weights in weight_list:
            diff = weights - mean
            prob = (torch.exp(-0.5 * ((diff / sigma) ** 2))/(sigma*torch.sqrt(torch.tensor(2*torch.pi)))).prod().item()
            probabilities.append(prob)
        normalized_probabilities = torch.tensor([prob / sum(probabilities) for prob in probabilities])
        return normalized_probabilities
    
    @staticmethod
    def open_data(data_path):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        return data
    
    @staticmethod
    def save_data(data,file_name):
        with open(file_name, "wb") as f:
            pickle.dump(data, f)