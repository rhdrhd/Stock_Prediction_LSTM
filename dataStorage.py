import pymongo
import datetime as dt
from bson import ObjectId
from dataAcquisition import *

MONGODB_URL = "mongodb+srv://root:yJka6Pp3Jz5G7Akk@cluster0.90czorv.mongodb.net/?retryWrites=true&w=majority"

def connect_to_mongoDB(url):
    """
    Connect to MongoDB
    
    Args:
        url (str): MongoDB URL
    
    Returns:
        pymongo.MongoClient: MongoDB client
    """
    client = pymongo.MongoClient(url)

    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)

    return client

def fetch_database(client, database_name):
    """
    Fetch a database from MongoDB
    
    Args:
        client (pymongo.MongoClient): MongoDB client
        database_name (str): Database name
    
    Returns:
        pymongo.database.Database: Database
    """

    if database_name in client.list_database_names():
        print(f"connecting to exisiting database {database_name}")
        return client[database_name]
    else:
        db = client[database_name]
        print(f"Created new database: {database_name}")
        return db
    
def fetch_collection(db, collection_name):
    """
    Fetch a collection in a database from MongoDB
    
    Args:
        db (pymongo.database.Database): Database
        collection_name (str): Collection name
    
    Returns:
        pymongo.collection.Collection: Collection
    """
    # Create a new collection
    if collection_name in db.list_collection_names():
        #print(f"connecting to existing collection {collection_name}")
        return db[collection_name]
    else:
        collection = db[collection_name]
        #print(f"Created new collection: {collection_name}")
        return collection




def create(collection, item)->bool:
    try:
        collection.insert_one(item)
        return True
    except Exception as e:
        print(e)
        return False

def read(collection, query:str):
    try:
        response = collection.find(query)
        return list(response)
        #return [item for item in response]
    except Exception as e:
        print(e)
        return None

def update(collection, item, properties: dict)->bool:
    try:
        collection.update_one(item, properties)
        return True
    except Exception as e:
        print(e)
        return False

def delete(collection, id:str)->bool:
    try:
        collection.delete_one({'_id':ObjectId(id)})
        return True
    except Exception as e:
        print(e)
        return False


def test_crud():
    client = connect_to_mongoDB(MONGODB_URL)
    db = fetch_database(client, "final_project")
    collection = fetch_collection(db, "combined_df")

    #read
    items = read(collection, {"Date": dt.datetime(2023, 3, 30)})
    print("the items read from mongodb: ", items)
    #create
    creation_flag = create(collection, {"Open": 300, "Close": 400, "Date": dt.datetime(2023, 3, 31)})
    print("creation flag: ", creation_flag)
    #check if it is created
    items = read(collection, {"Date": dt.datetime(2023, 3, 31)})
    print("the items read from mongodb after creation: ", items)
    #update
    update_flag = update(collection, {"Date": dt.datetime(2023, 3, 31)}, {"$set": {"Open": 100}})
    print("update flag: ", update_flag)
    #check if it is updated
    items = read(collection, {"Date": dt.datetime(2023, 3, 31)})
    #delete
    items = collection.find_one({"Date": dt.datetime(2023, 3, 31),"Open":100})
    delete_flag = delete(collection, items['_id'])
    print("delete flag: ", delete_flag)
    
def empty_database():
    client = connect_to_mongoDB(MONGODB_URL)
    db = fetch_database(client, "final_project")
    for collection_name in db.list_collection_names():
        collection = fetch_collection(db, collection_name)
        db.get_collection(collection_name).drop()
        print(f"{collection_name} dropped successfully")

def fetch_collection_from_db():
    """
    Fetch all collections from MongoDB

    Returns:
        dict: Dictionary containing all collections
    """
    data = {}
    client = connect_to_mongoDB(MONGODB_URL)
    db = fetch_database(client, "final_project")    
    for collection_name in db.list_collection_names():
        collection = fetch_collection(db, collection_name)
        collection_df = pd.DataFrame(list(collection.find()))
        collection_df = collection_df.drop(columns={"_id"})
        data[collection_name] = collection_df
    print("All data fetched successfully")
    return data

def insert_data_to_collection(df, collection):
    """
    Insert dataframe to a collection in MongoDB
    
    Args:
        df (pd.DataFrame): Dataframe
        collection (pymongo.collection.Collection): Collection
    """
    dict = df.to_dict('records')
    collection.insert_many(dict)

def store_data(data):
    """
    Store data to MongoDB
    
    Args:
        data (dict): Dictionary containing the data
    """
    # Connect to MongoDB
    client = connect_to_mongoDB(MONGODB_URL)
    db = fetch_database(client, "final_project")
    
    for key in data.keys():
        collection = fetch_collection(db, key)
        insert_data_to_collection(data[key], collection)

    print("All data stored successfully")









