# -*- coding: utf-8 -*-
# import json
# import math
# from typing import List, Dict, Any


# Task 1: 爬取所有商品資料
# -*- coding: utf-8 -*-
# import urllib2

import urllib.request
import urllib.error

def fetch_page():
    """抓取單一頁面的資料"""
    try:
        url = "https://24h.pchome.com.tw/store/DSAA31"
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Connection': 'keep-alive',
            'Host': '24h.pchome.com.tw'
        }

        # 建立請求物件
        req = urllib.request.Request(url, headers=headers)
        
        # 發送請求並接收回應
        with urllib.request.urlopen(req) as response:
            return response.read()

    except urllib.error.HTTPError as e:
        print(f"HTTP error occurred: {e.code} {e.reason}")
    except urllib.error.URLError as e:
        print(f"URL error occurred: {e.reason}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

    return None

def main():
    # 測試抓取第一頁
    data = fetch_page()
    if data:
        print("Successfully fetched page 1")
        print(data[:200])  # 只印出前 200 個字元來看看結果
    else:
        print("Failed to fetch data")

if __name__ == "__main__":
    main()

# # Task 2: 篩選高評價商品
# def get_best_products(products: List[Dict[str, Any]]) -> List[str]:
#     """
#     篩選出評分高於4.9且至少有1則評論的商品
#     Args:
#         products: 商品資料列表
#     Returns:
#         List[str]: 符合條件的商品ID列表
#     """
#     # TODO: 實作篩選邏輯
#     return []

# # Task 3: 計算特定商品平均價格
# def calculate_i5_average_price(products: List[Dict[str, Any]]) -> float:
#     """
#     計算Intel i5處理器電腦的平均價格
#     Args:
#         products: 商品資料列表
#     Returns:
#         float: 平均價格
#     """
#     # TODO: 實作計算邏輯
#     return 0.0

# # Task 4: 計算價格Z分數
# def calculate_price_z_scores(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """
#     計算所有商品價格的Z分數
#     Args:
#         products: 商品資料列表
#     Returns:
#         List[Dict]: 包含商品ID、價格和Z分數的列表
#     """
#     # TODO: 實作Z分數計算邏輯
#     return []

# def write_to_file(filename: str, data: List[str]) -> None:
#     """
#     將資料寫入檔案
#     Args:
#         filename: 檔案名稱
#         data: 要寫入的資料列表
#     """
#     with open(filename, 'w', encoding='utf-8') as f:
#         for item in data:
#             f.write(f"{item}\n")

# def write_to_csv(filename: str, data: List[Dict[str, Any]]) -> None:
#     """
#     將資料寫入CSV檔案
#     Args:
#         filename: 檔案名稱
#         data: 要寫入的資料列表
#     """
#     with open(filename, 'w', encoding='utf-8') as f:
#         f.write("ProductID,Price,PriceZScore\n")
#         for item in data:
#             f.write(f"{item['id']},{item['price']},{item['z_score']}\n")

# def main():
#     # 執行Task 1: 爬取所有商品
#     all_products = fetch_all_products()
#     product_ids = [p['id'] for p in all_products]
#     write_to_file('products.txt', product_ids)

#     # 執行Task 2: 篩選高評價商品
#     best_products = get_best_products(all_products)
#     write_to_file('best-products.txt', best_products)

#     # 執行Task 3: 計算i5處理器平均價格
#     i5_avg_price = calculate_i5_average_price(all_products)
#     print(f"Average price of ASUS PCs with Intel i5 processor: {i5_avg_price}")

#     # 執行Task 4: 計算價格Z分數
#     z_scores = calculate_price_z_scores(all_products)
#     write_to_csv('standardization.csv', z_scores)

# if __name__ == "__main__":
    # main()