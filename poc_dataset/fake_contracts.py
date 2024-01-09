import random
from typing import List

# Wallet to deposit and withdraw funds (ints) from. Invalid (bugged) transaction is when there is an integer underflow or overflow (not generated in generate_random_valid_transaction)
# Note: a negative amount is a valid input that is expected to throw an assertion.
class WalletSmartContract:
    def __init__(self):
        self.balance = 0
        self.max_int = 10 ** 7 # 2 ** 32
        self.normal_transaction_limit = 10 ** 7
        self.max_function_id = 1

    def deposit(self, amount: int) -> List:
        assert amount > 0
        self.balance += amount
        if self.balance >= self.max_int:
            self.balance = 0
        return {'success': 1}
    
    def withdraw(self, amount: int) -> List:
        assert amount > 0
        self.balance -= amount
        if self.balance < 0:
            self.balance = self.max_int - 1
        return {'success': 1}

    def parse_function_call(self, function_id, function_parameters):
        function_dispatcher = {0: self.deposit, 1: self.withdraw}
        unnormalized_function_parameters = self.unnormalize_function_parameters(function_parameters)
        return function_dispatcher[function_id](*list(unnormalized_function_parameters.values()))

    def get_name(self):
        return 'WalletSmartContract'
    
    def get_normalized_state(self):
        return {'balance': self.balance / self.max_int}
    
    def normalize_function_parameters(self, function_parameters):
        return {'amount': function_parameters['amount'] / self.normal_transaction_limit}
    
    def unnormalize_function_parameters(self, function_parameters):
        return {'amount': int(function_parameters['amount'] * self.normal_transaction_limit)}
    
    def is_transaction_positive(self, transaction):
        if transaction['function_parameters']['amount'] < 0 and transaction['transaction_return']['success'] == 0:
            return True
        if transaction['function_id'] == 0 / self.max_function_id and abs(transaction['starting_state']['balance'] * self.max_int + transaction['function_parameters']['amount'] * self.normal_transaction_limit - transaction['final_state']['balance'] * self.max_int) < 5:
            return True
        if transaction['function_id'] == 1 / self.max_function_id and abs(transaction['starting_state']['balance'] * self.max_int - transaction['function_parameters']['amount'] * self.normal_transaction_limit - transaction['final_state']['balance'] * self.max_int) < 5:
            return True
        return False

    def generate_random_positive_transaction(self):
        function_id = None
        function_parameters = {}
        if self.balance == 0:
            function_id = 0
            function_parameters = {'amount': random.randint(-self.normal_transaction_limit, min(self.max_int - self.balance - 1, self.normal_transaction_limit))}
            return function_id, self.normalize_function_parameters(function_parameters)
        elif self.balance == self.max_int - 1:
            function_id = 1
            function_parameters = {'amount': random.randint(-self.normal_transaction_limit, min(self.balance, self.normal_transaction_limit))}
            return function_id, self.normalize_function_parameters(function_parameters)

        if random.random() < 0.5:
            function_id = 0
            function_parameters = {'amount': random.randint(-self.normal_transaction_limit, min(self.max_int - self.balance - 1, self.normal_transaction_limit))}
            return function_id, self.normalize_function_parameters(function_parameters)
        else:
            function_id = 1
            function_parameters = {'amount': random.randint(-self.normal_transaction_limit, min(self.balance, self.normal_transaction_limit))}
            return function_id, self.normalize_function_parameters(function_parameters)
        
    def generate_random_unlabeled_transaction(self):
        function_id = None
        function_parameters = {}
        if random.random() < 0.5:
            function_id = 0
            function_parameters = {'amount': random.randint(-self.normal_transaction_limit, self.normal_transaction_limit)}
            return function_id, self.normalize_function_parameters(function_parameters)
        else:
            function_id = 1
            function_parameters = {'amount': random.randint(-self.normal_transaction_limit, self.normal_transaction_limit)}
            return function_id, self.normalize_function_parameters(function_parameters)