// This file uses CommonJS require instead of ES6 imports because it is not transpiled
const path = require('path');

module.exports = {
    entry: {
        sdfv: './src/sdfv.ts',
    },
    module: {
        rules: [
            {
                test: /\.m?[jt]sx?$/,
                use: [
                    {
                        loader: 'babel-loader',
                    },
                    {
                        loader: 'ts-loader',
                    },
                ],
                exclude: /node_modules/,
            },
        ],
    },
    resolve: {
        extensions: ['.ts', '.js'],
    },
    devtool: 'source-map',
    devServer: {
        static: {
            directory: __dirname,
        },
        devMiddleware: {
            writeToDisk: true,
        },
    },
    output: {
        filename: '[name].js',
        path: path.resolve(__dirname, 'dist'),
    },
};
