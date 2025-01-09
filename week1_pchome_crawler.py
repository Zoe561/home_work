# Task 1: 爬取所有商品資料
# -*- coding: utf-8 -*-
# import urllib2

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import urllib.error
import json
import sys

def fetch_products(page=1, page_count=40):
    """
    呼叫 PChome API，抓取指定頁碼的商品資料 (回傳 JSON)。
    page_count 預設 40，每頁抓多少可自行調整。
    回傳 Python dict，若抓不到則回傳 None。
    """
    base_url = (
        "https://ecshweb.pchome.com.tw/search/v4.3/all/results"
        "?cateid=DSAA31&attr=&pageCount={}&page={}"
    )
    url = base_url.format(page_count, page)

    # 設定 headers，模擬一般瀏覽器請求
    headers = {
            # 'Accept': '*/*',
            # 'Accept-Encoding': 'gzip, deflate, br, zstd',
            # 'Method': 'GET',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            # 'Connection': 'keep-alive',
            # 'referer': 'https://24h.pchome.com.tw/',
            }
    
    # 建立請求物件
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req) as response:
            # 讀取回傳資料 (可能是 gzip 壓縮)
            raw_data = response.read()
            encoding = response.info().get("Content-Encoding")
            if encoding == "gzip":
                import gzip
                raw_data = gzip.decompress(raw_data)
            
            # 將 JSON 文字轉為 Python dict
            data_dict = json.loads(raw_data.decode("utf-8", errors="ignore"))
            return data_dict
    except urllib.error.HTTPError as e:
        print(f"[HTTP Error] {e.code}: {e.reason}", file=sys.stderr)
    except urllib.error.URLError as e:
        print(f"[URL Error] {e.reason}", file=sys.stderr)
    except Exception as e:
        print(f"[Error] {str(e)}", file=sys.stderr)
    
    return None

def main():
    all_product_ids = set()
    page = 1
    page_count = 40

    while True:
        print(f"Fetching page {page} ...", file=sys.stderr)
        data_dict = fetch_products(page=page, page_count=page_count)
        
        if not data_dict:
            # 如果抓不到資料(回傳 None)，就結束
            break

        prods = data_dict.get("Prods", [])
        if not prods:
            # 沒有商品時，結束
            break
        
        # 擷取商品 Id
        for item in prods:
            pid = item.get("Id")
            if pid:
                all_product_ids.add(pid)
        
        # 若本頁抓到的商品數量 < page_count，表示可能已到最後一頁
        if len(prods) < page_count:
            break
        
        page += 1  # 下一頁

    # 輸出到 products.txt
    with open("products.txt", "w", encoding="utf-8") as f:
        for pid in sorted(all_product_ids):
            f.write(pid + "\n")
    
    print(f"Done! Total product IDs: {len(all_product_ids)}")
    print("All product IDs have been saved to 'products.txt'.")

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