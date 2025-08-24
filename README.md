# お祭り検索＆訪問記録アプリ

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/あなたのGitHubユーザ名/あなたのリポジトリ名/main/streamlit_app.py)

【注】このレポジトリは、Gemini CLI を使ってローカルディレクトリを自動的に修正し、その内容を GitHub にプッシュしたものです。

Cloud Runでのデプロイ先はこちら。
[https://hackathon-app-114970122290.asia-northeast1.run.app/](https://hackathon-app-114970122290.asia-northeast1.run.app/)

解説ブログ
[https://zenn.dev/chattso_gpt/articles/bbfeccaf9c71cf](https://zenn.dev/chattso_gpt/articles/bbfeccaf9c71cf)

Gemini APIとGoogle Drive/Spreadsheetを活用して、日本のお祭りを検索し、訪問記録を写真と共に管理できるStreamlitアプリケーションです。

## 主な機能

-   **お祭り検索**: Gemini APIのGoogle Search連携機能を使い、キーワード（例: 「東京 7月」）でお祭りを検索し、結果をスプレッドシートに自動で追加します。
-   **訪問記録の自動作成**: Google Driveにアップロードされた写真のGPS情報と撮影日時を基に、近くで開催されていたお祭りを自動で特定し、訪問記録を作成します。
-   **データの可視化**: 登録したお祭りの場所をマップに表示したり、訪問した場所のヒートマップを生成したりできます。
-   **統計情報**: お祭りの訪問率や総登録数などのサマリーをダッシュボードで確認できます。

## 必要なもの

-   Google Cloud Platformアカウント
-   Gemini APIキー
-   Python 3.8以上

## セットアップ手順

### 1. リポジトリをクローン

```bash
git clone https://github.com/あなたのGitHubユーザ名/あなたのリポジトリ名.git
cd あなたのリポジトリ名
```

### 2. Python仮想環境の作成と有効化

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. 必要なライブラリをインストール

```bash
pip install -r requirements.txt
```

### 4. Google Cloudの設定

このアプリケーションはGoogle SpreadsheetとGoogle Driveを使用します。以下の手順でAPIを有効にし、サービスアカウントを作成してください。

1.  [Google Cloud Console](https://console.cloud.google.com/)にアクセスします。
2.  新しいプロジェクトを作成します。
3.  **Google Drive API** と **Google Sheets API** を有効にします。
4.  **[認証情報]** > **[認証情報を作成]** > **[サービスアカウント]** を選択し、新しいサービスアカウントを作成します。
5.  作成したサービスアカウントに**「編集者」**のロールを付与します。
6.  サービスアカウントのキーを作成し、JSON形式でダウンロードします。
7.  Google Driveでフォルダを1つ作成し、そのフォルダIDを控えます。（URLの`folders/`以降の文字列）
8.  Google Spreadsheetでスプレッドシートを1つ作成し、そのIDを控えます。（URLの`d/`と`/edit`の間の文字列）
9.  作成したサービスアカウントのメールアドレス（`xxx@xxx.iam.gserviceaccount.com`）を、上記7のフォルダと8のスプレッドシートの**共有設定**に追加し、「編集者」権限を付与します。

### 5. 環境変数の設定

アプリケーションを実行するには、APIキーなどの秘密情報を環境変数として設定する必要があります。
`.env`という名前のファイルを作成し、以下のように記述してください。

```
# 4-6でダウンロードしたJSONファイルの中身を一行の文字列にして貼り付け
GOOGLE_SERVICE_ACCOUNT_JSON='{"type": "service_account", "project_id": "...", ...}'

# 4-8で控えたスプレッドシートのID
SPREADSHEET_ID="YOUR_SPREADSHEET_ID"

# 4-7で控えたGoogle DriveのフォルダID
DRIVE_FOLDER_ID="YOUR_DRIVE_FOLDER_ID"

# あなたのGemini APIキー
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```
**重要**: `.env`ファイルは絶対にGitで管理しないでください。`.gitignore`ファイルに`.env`が含まれていることを確認してください。

### 6. アプリケーションの実行

```bash
streamlit run streamlit_app.py
```

ブラウザで `http://localhost:8501` が自動的に開かれ、アプリケーションが表示されます。

## 使い方の例

1.  **お祭りを探す**: テキストボックスにキーワードを入力し、「検索」ボタンを押すと、Geminiがお祭りを検索して一覧に追加します。
2.  **訪問記録を更新**: スマートフォンのカメラアプリで位置情報(GPS)をオンにしてお祭りの写真を撮り、その写真をGoogle Driveの指定フォルダにアップロードします。「Google Driveの写真を処理して訪問記録を更新」ボタンを押すと、写真が自動で解析され、訪問記録が作成されます。
