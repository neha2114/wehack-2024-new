import { initializeAuthProxy } from '@propelauth/auth-proxy'

// Replace with your configuration
await initializeAuthProxy({
    authUrl: "https://75421095655.propelauthtest.com",
    integrationApiKey: "19235a5955a79425562e01ac48c45bc42fdbd7b211dcfc008c894620900951471e379481563756944c21432cf29f78a1",
    proxyPort: 8000,
    urlWhereYourProxyIsRunning: 'http://localhost:8000',
    target: {
        host: 'localhost',
        port: 8501,
        protocol: 'http:'
    },
})