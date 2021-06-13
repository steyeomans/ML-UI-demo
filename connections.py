"""
Data connection and transformation functions for the app.
"""

import psycopg2
import os
import pandas as pd
import sklearn
from sklearn import datasets
from dotenv import load_dotenv

load_dotenv()

class DB():
    "Creates database connection and querying options"
    def __init__(self):
        self.make_connection()

    def make_connection(self):
        "Initialize connection and cursor with autocommit"
        self.cnxn = psycopg2.connect(
            host = os.environ['AWS_DB_ENDPOINT'],
            port = os.environ['AWS_DB_PORT'],
            user = os.environ['AWS_DB_USER'],
            password = os.environ['AWS_DB_PWD'],
            database = os.environ['AWS_DB_NAME'],
            )
        self.cnxn.autocommit = True
        self.c = self.cnxn.cursor()

    def close(self):
        "Closes cursor and connection"
        try:
            self.c.close()
        except:
            print('Cursor already closed')
        try:
            self.cnxn.close()
        except:
            print('Connection already closed')

    def restart(self):
        self.close()
        self.make_connection()
        print('Restarted connection')

    def one(self, query, params=[]):
        "Execute a query and return a single row result"
        try:
            self.c.execute(query, params)
            result = self.c.fetchone()
        except psycopg2.errors.OperationalError:
            self.restart()
            self.c.execute(query, params)
            result = self.c.fetchone()
        return result

    def all(self, query, params=[]):
        "Execute a query and return a all rows results"
        try:
            self.c.execute(query, params)
            result = self.c.fetchall()
        except psycopg2.errors.OperationalError:
            self.restart()
            self.c.execute(query, params)
            result = self.c.fetchall()
        return result

    def none(self, query, params=[]):
        "Execute a query that returns no results"
        try:
            self.c.execute(query, params)
        except psycopg2.errors.OperationalError:
            self.restart()
            self.c.execute(query, params)

    def none_many(self, query, params=[]):
        "Execute a query that returns no results with execute_many"
        try:
            self.c.executemany(query, params)
        except psycopg2.errors.OperationalError:
            self.restart()
            self.c.executemany(query, params)
