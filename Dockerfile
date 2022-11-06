FROM nginx:1.10.1-alpine
COPY /vis/ /usr/share/nginx/html

CMD [ "nginx", "-g", "daemon off;"]
