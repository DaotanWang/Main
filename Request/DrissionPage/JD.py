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

        self.page.get('https://re.jd.com/search?keyword='+keyword+'&page='+num+'&enc=utf-8')
        self.page.wait.load_start()
        self.page.wait(3)

        elements = self.page.eles('@class=img_k')
        i = 0
        elements_price = self.page.eles('@class=price')
        price_list = [price1.text for price1 in elements_price]
        elements_common = self.page.eles('@class=praise praise-l')
        common_list = [common1.text for common1 in elements_common]

        for ele in elements:

            ele.click()
            lateat_tab = self.tab.latest_tab
            lateat_tab.wait(3)
            storename = lateat_tab.eles('@clstag=shangpin|keycount|product|dianpuname1')
            store_name = [nn.text for nn in storename][0]

            url = lateat_tab.url
            skuname = lateat_tab.eles('@class=sku-name')
            sku_name = [mm.text.replace('/',"") for mm in skuname][0]
            price = price_list[i]
            common = common_list[i].replace("已有", "").replace("人评价", "")
            from_1 = ""
            screenshot_path = os.path.join('tmp', f"{sku_name}.jpg")
            lateat_tab.get_screenshot(path='tmp', name=os.path.basename(screenshot_path), full_page=False)

            self.def_list = [store_name,sku_name,url,price,from_1,common,f"{sku_name}.jpg"]

            lateat_tab.close()
            sleep_time = random.uniform(1, 3)
            time.sleep(sleep_time)
            i= i + 1

            self.def_list_all=self.def_list_all + [self.def_list]

            if i >50:
                break
            else:
                pass

if __name__ == "__main__":
    platform_id = '1'
    scarp_name = '京东'
    scarp_method = '通过关键词搜索'
    keyword = ['军服']
    weibo = WeiboCrawler()
    weibo.search_weibo(keyword[0],'1')
    data = [['关键词','平台名称','店铺名','商品标题','价格','发货地','销量','首页截图']]

    # print(weibo.def_list)
    for j in range(0,50):
        data = data + [[keyword[0],scarp_name,weibo.def_list_all[j][0],weibo.def_list_all[j][1],weibo.def_list_all[j][3],weibo.def_list_all[j][4],weibo.def_list_all[j][5],weibo.def_list_all[j][6]]]

    print(data)
    with open('output2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


