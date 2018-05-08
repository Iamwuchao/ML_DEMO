
# encoding=utf-8
'''
通过迭代器获取mongo中的数据
'''
import pymongo


class Read_iterator:
    '''
        迭代器的初始化
    '''
    def __init__(self,batch_size,colection,query={}):
        self.__bathc_size = batch_size
        self.__colection = colection
        self.__query = query
        self.cursor = colection.find(query).batch_size(batch_size)

    def has_next(self):
        """ cursor alive? """
        if self.cursor and self.cursor.alive:
            return True
        else:
            if self.cursor.alive:
                self.cursor.close()
            return False

    def get_next(self):
        if self.has_next():
            return self.cursor.next()
        else:
            self.cursor.close()
            return None


'''
通过迭代器模式获取数据
'''
def read_traindata_iterator(batch_size,collection_name):
    db_client = pymongo.MongoClient("10.10.5.220", 27017)
    db = db_client.jindong
    collection = db[collection_name]
    read_iterator = Read_iterator(colection=collection,batch_size=batch_size,query={})
    return read_iterator