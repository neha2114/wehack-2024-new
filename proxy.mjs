import { initializeAuthProxy } from '@propelauth/auth-proxy'

// Replace with your configuration
await initializeAuthProxy({
    authUrl: "https://02386824.propelauthtest.com",
    integrationApiKey: "cb0444e05b66341d43c073097ee73bbc8d32fe935a743ed69a9f9c67b5edb3cc79aeeff4f0ea23156f95d3de6cef87c8",
    proxyPort: 8000,
    urlWhereYourProxyIsRunning: 'http://localhost:8000',
    target: {
        host: 'localhost',
        port: 8501,
        protocol: 'http:'
    },
})