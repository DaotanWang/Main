import os
import csv
from DrissionPage import ChromiumPage
from DrissionPage import Chromium
import time
import random
import re
class WeiboCrawler:
    def __init__(self):
        self.page = ChromiumPage()
        self.tab = Chromium()
        self.def_list = []
        self.def_list_all=[]

    def search_weibo(self, keyword,num):
        self.page.get('https://s.taobao.com/search?commend=all&ie=utf8&page='+num+'&q='+keyword+'&tab=all')
        self.page.wait.load_start()
        self.page.wait(3)

        elements = self.page.eles('@data-name=itemExp')
        i = 0
        elements_price_int = self.page.eles('@class:priceInt')
        price_int_list = [price1.text for price1 in elements_price_int]
        elements_price_float = self.page.eles('@class:priceFloat')
        price_float_list = [price1.text for price1 in elements_price_float]

        elements_common = self.page.eles('@class:realSales')
        common_list = [common1.text for common1 in elements_common]

        from_area = self.page.eles('@class:procity--')
        from_area_list = [from1.text for from1 in from_area]
        result_area_list = [from_area_list[i:i + 2] for i in range(0, len(from_area_list), 2)]

        for ele in elements:

            ele.click()
            lateat_tab = self.tab.latest_tab
            lateat_tab.wait(3)
            storename = lateat_tab.eles('@class:shopName')
            store_name = [nn.text for nn in storename][0]

            url = lateat_tab.url
            skuname = lateat_tab.eles('@class:mainTitle')
            sku_name = [mm.text.replace('/',"") for mm in skuname][0]
            price = f"{price_int_list [i]}{price_float_list[i]}"

            common = common_list[i].replace("人付款", "")

            from_2 = result_area_list[i]
            from_1 = from_2[0]+from_2[1]
            screenshot_path = os.path.join('tmp', f"{sku_name}.jpg")
            lateat_tab.get_screenshot(path='tmp', name=os.path.basename(screenshot_path), full_page=False)

            self.def_list = [store_name,sku_name,url,price,from_1,common,f"{sku_name}.jpg"]

            lateat_tab.close()
            sleep_time = random.uniform(1, 3)
            time.sleep(sleep_time)
            i= i + 1

            self.def_list_all=self.def_list_all + [self.def_list]

            if i >3:
                break
            else:
                pass

if __name__ == "__main__":
    platform_id = '1'
    scarp_name = '淘宝'
    scarp_method = '通过关键词搜索'
    keyword = ['军服']
    weibo = WeiboCrawler()
    weibo.search_weibo(keyword[0],'1')
    data = [['关键词','平台名称','店铺名','商品标题','价格','发货地','销量','首页截图']]

    # print(weibo.def_list)
    for j in range(0,2):
        data = data + [[keyword[0],scarp_name,weibo.def_list_all[j][0],weibo.def_list_all[j][1],weibo.def_list_all[j][3],weibo.def_list_all[j][4],weibo.def_list_all[j][5],weibo.def_list_all[j][6]]]

    print(data)
    with open('output2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


