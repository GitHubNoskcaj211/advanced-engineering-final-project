
from fake_contracts import WalletSmartContract
from copy import deepcopy
import pandas as pd
from tqdm import tqdm



def generate_data(original_smart_contract, generate_positive_data: bool, N_resets, N_transactions, name):
    all_data = []
    for simulation in tqdm(range(N_resets), desc='simulation'):
        smart_contract = deepcopy(original_smart_contract)
        for transaction in tqdm(range(N_transactions), desc='transaction'):
            starting_state = smart_contract.get_normalized_state()
            if generate_positive_data:
                function_id, function_parameters = smart_contract.generate_random_positive_transaction()
            else:
                function_id, function_parameters = smart_contract.generate_random_unlabeled_transaction()
            
            return_value = {}
            try:
                return_value = smart_contract.parse_function_call(function_id, function_parameters)
            except AssertionError:
                return_value = {'success': 0}
            final_state = smart_contract.get_normalized_state()
            transaction = {'contract': smart_contract.get_name(), 'function_id': function_id / smart_contract.max_function_id, 'function_parameters': function_parameters, 'starting_state': starting_state, 'final_state': final_state, 'transaction_return': return_value}
            transaction['positive'] = smart_contract.is_transaction_positive(transaction)
            all_data.append(transaction)

    df = pd.json_normalize(all_data, sep='.')
    # print(df[-df['positive']])
    if generate_positive_data:
        df.to_csv(f'poc_dataset/{original_smart_contract.get_name()}_poc_data_positive_{name}.csv', index=False)
    else:
        positive = df[df['positive']]
        negative = df[~df['positive']]
        positive.to_csv(f'poc_dataset/{original_smart_contract.get_name()}_poc_data_unlabeled_positive_{name}.csv', index=False)
        negative.to_csv(f'poc_dataset/{original_smart_contract.get_name()}_poc_data_unlabeled_negative_{name}.csv', index=False)

if __name__=='__main__':
    generate_data(WalletSmartContract(), True, 250, 10 ** 4, 'train')
    generate_data(WalletSmartContract(), False, 250, 10 ** 4, 'train')
    generate_data(WalletSmartContract(), False, 250, 10 ** 4, 'test')