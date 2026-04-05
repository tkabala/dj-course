FROM --platform=$BUILDPLATFORM node:22-bullseye-slim AS base
USER node
WORKDIR /home/node/app
COPY --chown=node:node package*.json ./
# Install oxc-parser native binding explicitly
RUN --mount=type=cache,uid=1000,gid=1000,target=/home/node/.npm \
    npm install --save-dev @oxc-parser/binding-linux-x64-gnu lightningcss-linux-x64-gnu @rollup/rollup-linux-x64-gnu || true && \
    npm install
COPY --chown=node:node . .

# --- Development Stage ---
FROM base AS development
ENV NODE_ENV=development
EXPOSE 3000
CMD ["npm", "run", "dev"]

# --- Build Stage ---
FROM base AS build
ENV NODE_ENV=production
RUN --mount=type=cache,uid=1000,gid=1000,target=/home/node/app/node_modules/.cache \
    npm run build

# --- Production Stage ---
FROM node:22-bullseye-slim AS production
USER node
WORKDIR /home/node/app
ENV NODE_ENV=production
ENV NITRO_HOST=0.0.0.0
ENV NITRO_PORT=3000

# Copy only built output and minimal dependencies
COPY --from=base /home/node/app/node_modules ./node_modules
COPY --from=build /home/node/app/.output ./.output

EXPOSE 3000
CMD ["node", ".output/server/index.mjs"]
