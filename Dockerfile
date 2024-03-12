FROM debian:bullseye-slim AS builder
 
COPY src /app/src
 
WORKDIR /app/src
 
RUN apt-get update && apt-get install -y \
    build-essential \
    pkgconf \
    automake \
    autoconf \
    autoconf-archive \
    libtool \
    libcurl4-openssl-dev \
    libicu-dev \
    libsqlite3-dev \
    gettext \
    autoconf \
    automake \
    libtool \
    g++ \
    libgtk-3-dev
 
WORKDIR /app/build
 
# Generate Makefiles
RUN ../src/autogen.sh
 
# Configure Freeciv
RUN ../src/configure --disable-nls
 
# Build Freeciv
RUN make
 
FROM debian:bullseye-slim
 
COPY --from=builder /app/build /app
 
COPY --from=builder /app/src/data /app
 
WORKDIR /app
 
RUN apt-get update && apt-get install -y \
    libtool \
    libcurl4-openssl-dev \
    libicu-dev \
    libsqlite3-dev \
    gettext \
    libtool \
    libgtk-3-dev
 
EXPOSE 5556
 
RUN useradd -ms /bin/bash user
 
USER user
 
CMD ["./fcser"]
 