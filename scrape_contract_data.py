import requests
from bs4 import BeautifulSoup
import pandas as pd
import io
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from tqdm import tqdm
from multiprocessing import Process
import cloudscraper
import undetected_chromedriver as uc
import random
from webdriver_manager.chrome import ChromeDriverManager
import os

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'}
CHROME_DRIVER_PATH = '/Users/jackson/Desktop/Classes/2023 Fall/Software Engineering/advanced-engineering-final-project/chromedriver'

def scrape_contract_transactions(contract_address, driver):
    SAVE_BREAK = 100
    
    page = 1
    all_data = pd.DataFrame(columns=['transaction_hash', 'block', 'from_address'])
    while True:
        url = f'https://etherscan.io/txs?a={contract_address}&ps=100&p={page}'
        driver.get(url)
        cc = 0
        while True:
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            if len(soup.find_all("div", {"class": "table-responsive"})) == 1:
                break
            time.sleep(0.1)
            cc += 1
            if cc > 4000:
                raise Exception('Can\'t get the table.')
        tables = soup.find_all("div", {"class": "table-responsive"})
        table_html = tables[0]
        table = pd.read_html(io.StringIO(str(table_html)))[0]
        table.rename(columns={'Txn Hash': 'transaction_hash', 'Block': 'block', 'From': 'from_address'}, inplace=True)
        if table.shape[0] == 1 and any(('There are no matching entries' in value for value in table['transaction_hash'])):
            break
        all_data = pd.concat([all_data, table[['transaction_hash', 'block', 'from_address']]], ignore_index=True)
        page += 1

        if page % SAVE_BREAK == 0:
            print('Starter scrape: ', contract_address, page)
            all_data.to_csv(f'dataset/{contract_address}_transactions_starter.csv', index=False)
        time.sleep(random.randint(8, 12))
    print('Done starter: ', contract_address)
    all_data.to_csv(f'dataset/{contract_address}_transactions_starter.csv', index=False)

def scrape_all_transaction_data(contract_address, driver):
    SAVE_BREAK = 10

    if not os.path.exists(f'dataset/{contract_address}_transactions_params.csv'):
        transactions = pd.read_csv(f'dataset/{contract_address}_transactions_starter.csv', low_memory=False)
        transactions['to_address'] = contract_address
        transactions['function'] = ''
        transactions['function_hash'] = 0
        transactions['transaction_return.success'] = ''
    else:
        transactions = pd.read_csv(f'dataset/{contract_address}_transactions_params.csv')

    for index, transaction in tqdm(list(transactions.iterrows()), desc=contract_address):
        if type(transaction['function_hash']) is str and '0x' in transaction['function_hash']:
            continue
        url = f'https://etherscan.io/tx/{transaction["transaction_hash"]}'
        driver.get(url)
        element = driver.find_element(By.ID, "ContentPlaceHolder1_collapseContent")
        driver.execute_script("arguments[0].setAttribute('class', 'collapse show')", element)

        cc = 0
        while True:
            soup_original = BeautifulSoup(driver.page_source, 'html.parser')
            if len(soup_original.find('div', {'id': 'ContentPlaceHolder1_collapseContent'}).find_all('textarea', {'id': 'inputdata'})) == 1:
                break
            time.sleep(0.1)
            cc += 1
            if cc > 4000:
                raise Exception('Can\'t get the table.', url)
            
        driver.execute_script("javascript:decodeInput();btnDecodeClick();")
        cc = 0
        while True:
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            if len(soup.find("div", {"id": "ContentPlaceHolder1_maintable"}).find_all(recursive=False)[1].find_all('table')) == 1:
                break
            time.sleep(0.1)
            cc += 1
            if cc > 4000:
                raise Exception('Can\'t get the table.', url)
        main_table = soup.find("div", {"id": "ContentPlaceHolder1_maintable"})
        general_data, extra_data, _ = main_table.find_all(recursive=False)
        main_table_original = soup_original.find("div", {"id": "ContentPlaceHolder1_maintable"})
        all_general_data = general_data.find_all(recursive=False)
        transactions.loc[index, 'transaction_return.success'] = 1 if any((element.text == 'Success' for element in all_general_data[1].find_all())) else 0
        transactions.loc[index, 'from_address'] = all_general_data[9].find('span', {'class' : 'me-1'}).get_text()

        _, extra_data_original, _ = main_table_original.find_all(recursive=False)
        input_data = extra_data_original.find('div', {'id': 'ContentPlaceHolder1_collapseContent'}).find('textarea', {'id': 'inputdata'}).text
        input_data_split = input_data.split('\n')
        transactions.loc[index, 'function_hash'] = input_data_split[2].split(': ')[-1]
        transactions.loc[index, 'function'] = input_data_split[0].split(': ')[-1]
        
        function_parameters = pd.read_html(io.StringIO(str(extra_data.find('div', {'id': 'inputDecode'}).find('table'))))[0]
        for _, parameter in function_parameters.iterrows():
            if f'function_parameters.{parameter["Name"]}.{parameter["Type"]}' not in transactions.columns:
                transactions[f'function_parameters.{parameter["Name"]}.{parameter["Type"]}'] = ''
            transactions.loc[index, f'function_parameters.{parameter["Name"]}.{parameter["Type"]}'] = parameter['Data']
        if index % SAVE_BREAK == 0:
            transactions.to_csv(f'dataset/{contract_address}_transactions_params.csv', index=False)
        time.sleep(random.randint(8, 12))

    transactions.to_csv(f'dataset/{contract_address}_transactions_params.csv', index=False)

def fully_process(contract, startup_time):
    print('Starting: ', contract)
    time.sleep(startup_time)
    driver = uc.Chrome()
    # scrape_contract_transactions(contract, driver)
    scrape_all_transaction_data(contract, driver)
    driver.quit()
    print('COMPLETE: ', contract)

if __name__ == '__main__':
    # fully_process('0xc5d105e63711398af9bbff092d4b6769c82f793d', 0)
    # input()
    tasks = [
        (fully_process, '0xc5d105e63711398af9bbff092d4b6769c82f793d', 0),
        (fully_process, '0x27f706edde3aD952EF647Dd67E24e38CD0803DD6', 1),
        (fully_process, '0xb75a5e36cc668bc8fe468e8f272cd4a0fd0fd773', 2),
        (fully_process, '0x330bebabc9a2a4136e3d1cb38ca521f5a95aec2e', 3),
        (fully_process, '0xB5335e24d0aB29C190AB8C2B459238Da1153cEBA', 4),
        (fully_process, '0x103c3a209da59d3e7c4a89307e66521e081cfdf0', 5),
        (fully_process, '0xf084d5bc3e35e3d903260267ebd545c49c6013d0', 6),
        # (fully_process, '0x55F93985431Fc9304077687a35A1BA103dC1e081', 7),
        # (fully_process, '0x78b7fada55a64dd895d8c8c35779dd8b67fa8a05', 8),
        # (fully_process, '0x219218f117dc9348b358b8471c55a073e5e0da0b', 9),
        # (fully_process, '0x00a0cbe98e4d110b0fa82646152d77babf2951d0', 10),
        # (fully_process, '0x6aac8cb9861e42bf8259f5abdc6ae3ae89909e11', 11),
        # (fully_process, '0x9653cFd0865ad8313BEA2f0C2EC0584BFd05115B', 12),
        # (fully_process, '0x0bb217e40f8a5cb79adf04e1aab60e5abd0dfc1e', 13),
        # (fully_process, '0x0b76544f6c413a555f309bf76260d1e02377c02a', 14),
    ]
    assert len(set([address for _, address, _ in tasks])) == len(tasks), 'Duplicate contract address - code is not safe.'
    procs = []
    for function, address, startup_time in tasks:
        proc = Process(target=function, args=(address, startup_time))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
    print('All Done.')