# dva-uts-project
    DVA UTS Group Project

## Instructions

_citing_: https://github.com/mcekovic/tennis-crystal-ball/issues
Ultimate Tennis Statistics and Tennis Crystal Ball pre-populated database with ATP tennis data as of season 2021.

Run the container as:

docker run -p 5432:5432 -d --name uts-database mcekovic/uts-database

Enter PostgreSQL client command line tool:

docker exec -it uts-database psql -U tcb

Execute SQL commands, like:

SELECT goat_rank, name, country_id, goat_points FROM player_v ORDER BY goat_points DESC NULLS LAST LIMIT 20;

To exit psql command line tool: quit

To stop Docker container running UTS database: docker stop uts-database

- The original database: https://github.com/JeffSackmann/tennis_atp