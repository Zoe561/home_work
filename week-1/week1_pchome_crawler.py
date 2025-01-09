#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import urllib.error
import json
import sys
import gzip
import math

def fetch_products(page=1, page_count=40):

    base_url = (
        "https://ecshweb.pchome.com.tw/search/v4.3/all/results"
        "?cateid=DSAA31&attr=&pageCount={}&page={}"
    )
    url = base_url.format(page_count, page)

    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Encoding": "gzip, deflate, br",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Connection": "keep-alive",
        "Referer": "https://24h.pchome.com.tw/"
    }

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req) as response:
            raw_data = response.read()
            if response.info().get("Content-Encoding") == "gzip":
                raw_data = gzip.decompress(raw_data)

            data_dict = json.loads(raw_data.decode("utf-8", errors="ignore"))
            return data_dict
    except urllib.error.HTTPError as e:
        print(f"[HTTP Error] {e.code}: {e.reason}", file=sys.stderr)
    except urllib.error.URLError as e:
        print(f"[URL Error] {e.reason}", file=sys.stderr)
    except Exception as e:
        print(f"[Error] {str(e)}", file=sys.stderr)

    return None


def task1_fetch_all_products():
    """
    逐頁抓取 DSAA31 分類的商品，並回傳「所有商品」的列表。
    另外也把商品 Id 寫入 products.txt (每行一個)。
    回傳: List[Dict]，其中每個元素包含商品的關鍵資訊。
    """
    all_products = []
    all_ids = []
    page = 1
    page_count = 40

    while True:
        print(f"Fetching page {page}...", file=sys.stderr)
        data = fetch_products(page=page, page_count=page_count)
        if not data:
            break

        prods = data.get("Prods", [])
        if not prods:
            break

        all_products.extend(prods)

        # 收集商品 ID
        for item in prods:
            pid = item.get("Id")
            if pid:
                all_ids.append(pid)

        if len(prods) < page_count:
            # 商品數量少於 page_count，應該是最後一頁
            break

        page += 1

    # 將所有商品 ID 寫入 products.txt
    with open("products.txt", "w", encoding="utf-8") as f:
        for pid in all_ids:
            f.write(pid + "\n")

    print(f"Task 1 done. Total products fetched: {len(all_products)}")
    return all_products


def task2_best_products(products):
    """
    根據 Task 1 抓到的 products (List[Dict])，
    篩選出 reviewCount >= 1 & averageRating > 4.9 的商品，
    將其 ID 寫入 best-products.txt (每行一個)。
    """

    
    best_ids = []
    for p in products:

        review_count = p.get("reviewCount", 0)
        rating_value = p.get("ratingValue", 0)
        pid = p.get("Id", "")
        if review_count is None:
            review_count = 0

        if rating_value is None:
            rating_value = 0
        
        # 檢查是否符合條件
        if review_count >= 1 and rating_value > 4.9:
            best_ids.append(pid)

    # 寫檔
    with open("best-products.txt", "w", encoding="utf-8") as f:
        for pid in best_ids:
            f.write(pid + "\n")

    print(f"Task 2 done. Found {len(best_ids)} products with ratingValue > 4.9 & reviewCount >= 1.")


def task3_average_price_of_i5(products):
    """
    從 Task 1 的資料中，挑出商品名稱中含 'i5' 來判斷。
    """
    total_price = 0
    count = 0

    for p in products:
        name = p.get("Name", "").lower()
        price = p.get("Price", 0)

        # 檢查商品名稱是否包含 'i5'
        if "i5" in name:
            total_price += price
            count += 1

    if count > 0:
        avg_price = total_price / count
        print(f"Task 3: The average price of ASUS PCs with Intel i5 is: {avg_price:.2f}")
    else:
        print("Task 3: No ASUS i5 products found, can't calculate average price.")

def task4_standardize_prices(products):
    """
    Task 4:
    使用 z-score 來標準化所有 ASUS PC 價格 (視整個 products 為母體)。
    1. 計算母體平均 (mu) 和 母體標準差 (sigma)。
    2. 逐一計算 z-score = (price - mu) / sigma。
    3. 將結果寫入 standardization.csv，格式:
       ProductID,Price,PriceZScore
    """
    # 先收集所有有價格的商品 (price 不為 None)，順便記下 ProductID
    valid_items = []
    for p in products:
        pid = p.get("Id")
        price = p.get("Price")
        # 只收錄確實有價格的商品
        if pid and isinstance(price, (int, float)):
            valid_items.append((pid, price))

    if not valid_items:
        print("Task 4: No valid price data to calculate Z-Score.")
        return

    # 取出所有價格
    prices = [item[1] for item in valid_items]

    # 計算母體平均 mu
    mu = sum(prices) / len(prices)

    # 計算母體標準差 sigma
    # population variance = sum((x - mu)^2) / N
    # population stdev = sqrt(variance)
    variance = sum((x - mu) ** 2 for x in prices) / len(prices)
    sigma = math.sqrt(variance)

    # 如果 sigma == 0，代表所有價格都相同，z-score 全部是 0
    if sigma == 0:
        print("Task 4: All prices are the same. Z-Score will be 0 for all.")
        sigma = 1  # 避免除以 0

    # 計算 z-score
    z_data = []
    for pid, price in valid_items:
        z_score = (price - mu) / sigma
        z_data.append((pid, price, z_score))

    # 寫入 CSV
    with open("standardization.csv", "w", encoding="utf-8") as f:
        # 表頭
        f.write("ProductID,Price,PriceZScore\n")
        # 逐行寫入
        for pid, price, z_score in z_data:
            # 格式: ProductID,Price,PriceZScore
            f.write(f"{pid},{price},{z_score:.5f}\n")

    print(f"Task 4 done. Wrote {len(z_data)} lines to standardization.csv.")

def main():
    # Task 1: 抓取所有商品 (並輸出 products.txt)
    products = task1_fetch_all_products()

    # Task 2: 篩選高評分高評論商品 -> best-products.txt
    task2_best_products(products)

    # Task 3: 計算搭載 i5 處理器的平均售價 -> 直接印在 console
    task3_average_price_of_i5(products)

    # Task 4: 以整個 products 為母體，計算價格 z-score -> standardization.csv
    task4_standardize_prices(products)

if __name__ == "__main__":
    main()
