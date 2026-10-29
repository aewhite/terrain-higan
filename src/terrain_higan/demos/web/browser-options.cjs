function browserOptions(options = {}) {
  return { executablePath: process.env.CHROME_PATH || undefined, ...options };
}

module.exports = { browserOptions };
