from DrissionPage import ChromiumPage
import logging

class XiaohongshuCrawler:
    def __init__(self):
        self.page = ChromiumPage()

    def is_login(self):
        self.page.get('http://www.xiaohongshu.com/explore')
        lis = self.page.eles('css:ul.channel-list li')
        # logging.info(lis)
        count = len(lis)
        # logging.info(count)
        if count == 3:
            self.isLogin = False
            self.login()
        elif count == 4:
            self.isLogin = True
            logging.info(f'登录成功')
        else:
            logging.info(f'li 的数量不是 3 和 4 无法判定是否登录')

    def login(self):
        input('请扫码登录，登录成功后敲击回车')
        self.isLogin = True

    def input_keyword(self,keyword):
        self.page.ele('css:#search-input').input(keyword)
        self.page.actions.key_down('Enter')
        self.page.wait.load_start()
        self.page.wait(3)

    def get_names(self):
        elements = self.page.eles('css:.name')
        names = [element.text for element in elements]
        return names

    def run(self):
        self.is_login()
        self.input_keyword(keyword)

if __name__ == '__main__':
    keyword = "华为智慧屏"
    xhs = XiaohongshuCrawler()
    xhs.run()
    names = xhs.get_names()
    for name in names:
        print(name)
