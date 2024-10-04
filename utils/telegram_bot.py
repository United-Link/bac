# -*- coding:utf-8 _*-  
import requests
from typing import Any, Union, Optional

class TgBot:
    def __init__(self, token: Any, telegram_group_id: Optional[Union[str, int]] = None):
        self.token = token
        self.send_photo_url = "https://api.telegram.org/bot{}/sendPhoto".format(token)
        self.telegram_group_id = telegram_group_id

    def send_message(self, telegram_group_chat_id: Optional[Union[str, int]] = None,
                     text: Optional[Any] = None, parse_mode: Optional[Any] = None) -> requests.Response:
        """
        Send message from telegram bot in group
        """
        if telegram_group_chat_id is None:
            telegram_group_chat_id = self.telegram_group_id

        request = "https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}".format(self.token,
                                                                                         telegram_group_chat_id,
                                                                                         text)
        if parse_mode is not None:
            request += "&parse_mode={}".format(parse_mode)
        r = requests.get(request)
        return r

    def send_photo(self, image_io, telegram_group_chat_id: Optional[Union[str, int]] = None,
                   caption: Optional[str] = None) -> requests.Response:
        """
        Send photo from telegram bot in group
        :param image_io: 
        :param telegram_group_chat_id
        :param caption
        :return: response
        """
        if telegram_group_chat_id is None:
            telegram_group_chat_id = self.telegram_group_id

        data = {'chat_id': telegram_group_chat_id}
        if caption:
            data["caption"] = caption
        r = requests.post(self.send_photo_url, files={'photo': image_io}, data=data)
        return r