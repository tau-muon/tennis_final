import psycopg2
from pandas import DataFrame
import pandas.io.sql as psql


class Database(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(Database, cls).__new__(cls, *args, **kwargs)
            cls.connection = psycopg2.connect(user = "tcb", password = "tcb", host = "localhost", port = "5432", database = "tcb")
            cls.cursor = cls.connection.cursor()
        return cls._instance


    def getdf(self, tableName:str, sqlquery:str="") -> DataFrame:
        if sqlquery == "":
            sqlquery = 'SELECT * FROM {tableName}'.format(tableName=tableName)
        return psql.read_sql(sqlquery, self.connection)
    
    
    def getdf_tablenames(self) -> DataFrame:
        return psql.read_sql("SELECT * FROM INFORMATION_SCHEMA.TABLES GO ", self.connection)


    def execut_sql(self, query:str) -> list:
        self.cursor.execute(query)
        return self.cursor.fetchall()


    def close(self):
        self.cursor.close()
        self.connection.close()
        return