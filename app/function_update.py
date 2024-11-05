import flask
import functions_framework
from google.cloud.sql.connector import Connector, IPTypes
import os
import pg8000
import sqlalchemy


db = None


def connect_with_connector() -> sqlalchemy.engine.base.Engine:
    """
    Initializes a connection pool for a Cloud SQL instance of Postgres.

    Uses the Cloud SQL Python Connector package.
    """
    # Note: Saving credentials in environment variables is convenient, but not
    # secure - consider a more secure solution such as
    # Cloud Secret Manager (https://cloud.google.com/secret-manager) to help
    # keep secrets safe.

    # e.g. 'project:region:instance'
    instance_connection_name = os.environ["INSTANCE_CONNECTION_NAME"]
    db_user = os.environ["DB_USER"]  # e.g. 'my-db-user'
    db_pass = os.environ["DB_PASS"]  # e.g. 'my-db-password'
    db_name = os.environ["DB_NAME"]  # e.g. 'my-database'

    ip_type = IPTypes.PRIVATE if os.environ.get(
        "PRIVATE_IP") else IPTypes.PUBLIC

    # initialize Cloud SQL Python Connector object
    connector = Connector()

    def getconn() -> pg8000.dbapi.Connection:
        conn: pg8000.dbapi.Connection = connector.connect(
            instance_connection_name,
            "pg8000",
            user=db_user,
            password=db_pass,
            db=db_name,
            ip_type=ip_type,
        )
        return conn

    # The Cloud SQL Python Connector can be used with SQLAlchemy
    # using the 'creator' argument to 'create_engine'
    pool = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
        # [START_EXCLUDE]
        # Pool size is the maximum number of permanent connections to keep.
        pool_size=5,
        # Temporarily exceeds the set pool_size if no connections are available.
        max_overflow=2,
        # The total number of concurrent connections for your application will be
        # a total of pool_size and max_overflow.
        # 'pool_timeout' is the maximum number of seconds to wait when retrieving a
        # new connection from the pool. After the specified amount of time, an
        # exception will be thrown.
        pool_timeout=30,  # 30 seconds
        # 'pool_recycle' is the maximum number of seconds a connection can persist.
        # Connections that live longer than the specified amount of time will be
        # re-established
        pool_recycle=1800,  # 30 minutes
        # [END_EXCLUDE]
    )
    return pool


def init_connection_pool() -> sqlalchemy.engine.base.Engine:
    """Sets up connection pool for the app."""
    # use a TCP socket when INSTANCE_HOST (e.g. 127.0.0.1) is defined
    return connect_with_connector()


@functions_framework.http
def hello_http(request: flask.Request) -> flask.Response:
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    global db
    # initialize db within request context
    if not db:
        # initiate a connection pool to a Cloud SQL database
        db = init_connection_pool()
        # creates required 'votes' table in database (if it does not exist)

    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'name' in request_json:
        name = request_json['name']
    elif request_args and 'name' in request_args:
        name = request_args['name']
    else:
        name = 'World'

    results = []

    with db.connect() as conn:
        # Execute the query and fetch all results
        embedding_model = os.environ["EMBEDDING_MODEL"]
        if embedding_model == "textembedding-gecko@001":
            embedding_column = "amenities_embedding.embedding_textembedding_gecko001"
        elif embedding_model == "text-embedding-004":
            embedding_column = "amenities_embedding.embedding_textembedding_004"
        else:
            embedding_column = "amenities_embedding.embedding"
# Parameterize this to pull embedding column from variable rather than hardcoded
# Don't spend more than 30 minutes. This lab is DONE. THIS ISN'T REQUIRED
        search_results = conn.execute(
            sqlalchemy.text(
                "SELECT name, description, location "  # terminal, category, hour
                "FROM amenities "
                "JOIN amenities_embedding ON amenities.id = amenities_embedding.id "
                "ORDER BY amenities_embedding.embedding_textembedding_004 "
                "<-> embedding(:embedding_model, :search_term)::vector "
                "LIMIT 3 "
            ),
        parameters={"embedding_model": embedding_model,
                    "search_term": name}).fetchall()
        # Convert the results into a list of dicts representing votes
        for row in search_results:
            results.append(
                {"name": row[0], "description": row[1], "location": row[2]})

    return {
        "results": results,
    }
