import csv
import re
from urllib.parse import urlparse
from typing import Dict, List, Any


def calculate_attack_weight(request):
    """
    Calculates the attack weight for a given HTTP request based on the features
    described in the research paper.

    Args:
        request (dict): A dictionary representing the parsed HTTP request,
                        containing keys for 'url', 'payload', 'headers', and 'files'.
                        The structure of 'files' is expected to be a list of file paths
                        or file objects that can be processed.

    Returns:
        int: The calculated attack weight.
    """

    url = request.get('url', '')
    payload = request.get('payload', '')
    headers = request.get('headers', {})
    files = request.get('files', [])

    # 1. Calculate URL Weight (u)
    url_weight = calculate_url_weight(url)

    # 2. Calculate Number of Attack Words in Inputs (v)
    attack_words_weight = calculate_attack_words_weight(payload, headers)

    # 3. Calculate Manipulate Payload Weight (m)
    manipulation_weight = calculate_manipulation_weight(payload, headers)

    # 4. Calculate Alphanumeric Character to Special Character Ratio (r)
    ratio_weight = calculate_ratio_weight(payload)

    # 5. Calculate Files Weight (F)
    files_weight = calculate_files_weight(files)

    # 6. Calculate Final Attack Weight (z)
    attack_weight = url_weight + attack_words_weight + manipulation_weight + ratio_weight + files_weight
    return attack_weight


def calculate_url_weight(url):
    """
    Calculates the URL weight based on the presence of malicious elements.

    Args:
        url (str): The absolute URL of the request.

    Returns:
        int: The URL weight.
    """
    url_weight = 0
    parsed_url = urlparse(url)
    path = parsed_url.path

    # --- Malicious URL Patterns ---
    # More comprehensive list targeting path and query parameters.
    malicious_url_patterns = {
        # --- Sensitive File/Directory Access ---
        r"/\.env": 250,                      # Environment config
        r"/\.git(/|\b|$)": 220,               # Git repo data (broader match)
        r"/\.svn(/|\b|$)": 200,               # Subversion repo data
        r"/\.hg(/|\b|$)": 200,                # Mercurial repo data
        r"/\.bzr(/|\b|$)": 200,               # Bazaar repo data
        r"/\.DS_Store": 100,                 # macOS metadata file (can leak filenames)
        r"/WEB-INF/": 180,                   # Java EE sensitive directory
        r"/META-INF/": 170,                   # Java sensitive directory
        r"web\.config": 190,                 # IIS config file
        r"/\.htpasswd": 200,                 # Apache password file
        r"/\.htaccess": 150,                 # Apache config file
        r"/etc/passwd": 200,                 # *nix password file
        r"/etc/shadow": 280,                 # *nix shadow password file
        r"/etc/group": 150,                  # *nix group file
        r"/etc/hosts": 100,                  # Host file
        r"/proc/self/environ": 260,          # Linux process environment variables
        r"/proc/version": 150,               # Linux kernel version info
        r"/proc/cmdline": 170,               # Linux kernel boot parameters
        r"/root/\.bash_history": 270,        # Root's command history
        r"/\.bash_history": 180,             # User's command history
        r"/\.ssh/id_rsa": 300,               # SSH private key
        r"/\.ssh/authorized_keys": 200,      # SSH authorized keys
        r"/\.aws/credentials": 300,          # AWS credentials
        r"/\.azure/accessTokens\.json": 300, # Azure credentials
        r"/\.dockerenv": 150,                # Docker environment indicator
        r"/server-status": 120,              # Apache server status page
        r"/server-info": 120,                # Apache server info page
        r"/phpinfo\.php": 160,               # PHP info disclosure
        r"/info\.php": 150,                  # Common PHP info file name
        r"\b(config|settings|credential|secret|password|key|token|connectionstring)\b.*\.(json|yaml|yml|xml|ini|php|bak|old|swp|txt|cfg)\b": 200, # Broader config/secret file pattern
        r"\.(bak|backup|old|tmp|temp|swp|~)$": 130, # Common backup/temporary file extensions
        r"/logs?/": 80,                      # Accessing log directories
        r"\.log$": 90,                       # Accessing log files directly
        r"access\.log": 100,                 # Common access log name
        r"error\.log": 100,                  # Common error log name
        r"/wp-config\.php": 200,             # WordPress config
        r"/configuration\.php": 190,         # Joomla config
        r"/sites/default/settings\.php": 190,# Drupal config
        r"/app/etc/local\.xml": 210,         # Magento 1 config
        r"/app/etc/env\.php": 210,           # Magento 2 config

        # --- Directory Traversal / File Inclusion ---
        r"\.\./": 150,                       # Basic traversal (weight should increase with frequency)
        r"%2e%2e%2f": 160,                   # URL encoded ../
        r"%2e%2e%5c": 160,                   # URL encoded ..\
        r"\.\.%c0%af": 180,                  # UTF-8 overlong encoding ../
        r"\.\.%c1%9c": 180,                  # UTF-8 overlong encoding ..\
        r"\.\.+": 170,                       # Multiple dots (potential filter bypass)
        r"(%252e){2}%252f": 190,             # Double URL encoded ../
        r"php://filter": 280,                # PHP Filter LFI/RFI vector
        r"php://input": 250,                 # PHP Input RCE vector
        r"expect://": 260,                   # Expect RCE vector
        r"data:text/plain": 180,             # Data wrapper RFI/XSS vector
        r"file:///": 250,                    # Local file scheme access
        r"(\?|&)page=.*\.\.": 160,           # Traversal in common 'page' parameter
        r"(\?|&)include=.*\.\.": 160,        # Traversal in common 'include' parameter
        r"(\?|&)file=.*\.\.": 160,           # Traversal in common 'file' parameter
        r"(\?|&)path=.*\.\.": 160,           # Traversal in common 'path' parameter
        r"(\?|&)document=.*\.\.": 160,       # Traversal in common 'document' parameter

        # --- Command Injection ---
        # Parameters often abused
        r"(\?|&)(cmd|exec|query|run|command|shell|ping|cmdexec|exe|do|arg|argument|input|data|url|uri|download|upload|file|filename|path|payload|data)=.*(&|$)": 200,
        # Common commands within parameters (needs payload check too, but can indicate intent in URL)
        r"(\?|&).*(wget|curl|bash|sh|nc|netcat|powershell|cmd\.exe|;|\||`|\$\(|\$\{)": 220,

        # --- SQL Injection ---
        # Basic patterns in query params
        r"(\?|&).*(%27|'|%3b|;)\s*(and|or|xor)\b": 180, # Basic boolean SQLi probes (' OR, ' AND)
        r"(\?|&).*(\bunion\b.{1,50}?\bselect\b)": 250, # UNION SELECT (case-insensitive, allowing space/chars)
        r"(\?|&).*(select\s*@@version|select\s*version\(\)|select\s*user\(\)|select\s*database\(\)|select\s*db_name\(\))": 200, # Common info gathering
        r"(\?|&).*(information_schema|pg_catalog|sys\.databases)": 190, # Schema querying
        r"(\?|&).*(\bwaitfor\b\s*delay\b|\bsleep\(|\bpq_sleep\(|\bdbms_lock\.sleep\()": 260, # Time-based SQLi functions
        r"(\?|&).*(benchmark\()": 240,           # Time-based SQLi (BENCHMARK)
        r"(\?|&).*(--|\#|/\*)": 100,             # SQL comments (can have FPs)

        # --- XSS ---
        # Basic patterns in query params (Payload check is usually more effective)
        r"(\?|&).*(<script>|<img|<svg|<iframe|<body|<style|<link|<meta)": 150, # HTML tags often used in XSS
        r"(\?|&).*(alert\(|confirm\(|prompt\(|document\.cookie|window\.location|eval\(|expression\(|javascript:|vbscript:)": 170, # XSS keywords/functions
        r"(\?|&).*(onerror=|onload=|onmouseover=|onclick=|onfocus=|onblur=|oninput=)": 180, # Event handlers

        # --- Server-Side Request Forgery (SSRF) ---
        r"(\?|&).*(127\.0\.0\.1|localhost|\[::1\])": 100, # Targeting localhost (can be legitimate, lower weight)
        r"(\?|&).*169\.254\.169\.254": 300,         # AWS metadata service IP
        r"(\?|&).*metadata\.google\.internal": 300, # GCP metadata service hostname
        r"(\?|&).*168\.63\.129\.16": 300,           # Azure metadata service IP
        r"(\?|&).*instance-data/latest/meta-data": 290, # Oracle Cloud metadata path
        r"(\?|&).*(10\.\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3})": 120, # Internal IP ranges (can have FPs)
        r"(\?|&).*(file:///|dict://|gopher://|ldap://|sftp://)": 270, # Protocols often abused in SSRF

        # --- Scanning / Tooling Signatures ---
        r"/phpmyadmin(/|\b|$)": 80,
        r"/pma(/|\b|$)": 80,
        r"/adminer": 90,
        r"/wp-admin(/|\b|$)": 60,
        r"/wp-login\.php": 60,
        r"/jmx-console": 150,
        r"/web-console": 140,
        r"/solr(/|\b|$)": 80,
        r"/elasticsearch(/|\b|$)": 90,
        r"/_cat/indices": 100, # Elasticsearch info endpoint
        r"/\.git/config": 250,
        r"/cgi-bin/": 70,
        r"/scripts/": 70,
        r"/cfide/": 120, # ColdFusion directory
        r"/\.well-known/": 30, # Often legit, but sometimes scanned for specific vulns

        # --- API / Framework Specific ---
        r"/graphql": 100,                   # Common GraphQL endpoint
        r"/v[1-9]/": 50,                     # Versioned API path (can be noisy)
        r"/api/": 40,                        # Generic API path (can be noisy)
        r"/_profiler/": 180,                # Symfony profiler path
        r"/_ignition/": 190,                # Laravel Ignition debug path
        r"/actuator/": 170,                 # Spring Boot Actuator path
        r"/boaform/admin/formLogin": 150,   # Common router login path

        # --- Log4j / JNDI Injection ---
        r"(\?|&).*(jndi:(ldap|rmi|dns)|%24%7Bjndi:)": 350, # JNDI injection patterns (case insensitive recommended)

    }

    for pattern, weight in malicious_url_patterns.items():
        if re.search(pattern, path, re.IGNORECASE):
            url_weight += weight
    return url_weight


def calculate_attack_words_weight(payload, headers):
    """
    Calculates the weight based on the presence of attack words in payload and headers.

    Args:
        payload (str): The request payload.
        headers (dict): The request headers.

    Returns:
        int: The attack words weight.
    """
    attack_words_weight = 0
    inputs = payload + " ".join(headers.values())

    # --- Attack Words ---
    # Keywords/patterns often found in request payloads (form data, JSON, XML, etc.) or headers.
    # Apply case-insensitive matching and consider decoding inputs (URL, HTML entities).

    attack_words = {
        # --- SQL Injection ---
        "union select": 250,
        "union all select": 250,
        "union distinct select": 250,
        "order by": 30,                     # Reduced from 80
        "group by": 30,                     # Reduced from 70
        "having": 100,
        "information_schema": 200,
        "pg_catalog": 200,
        "sys.databases": 190,
        "sys.tables": 190,
        "sys.columns": 190,
        "@@version": 180,
        "version()": 180,
        "user()": 50,                       # Reduced from 170
        "database()": 170,
        "db_name()": 170,
        "current_user": 50,                 # Reduced from 170
        "session_user": 50,                 # Reduced from 170
        "system_user": 50,                  # Reduced from 170
        "schema_name": 170,
        "load_file(": 250,
        "outfile": 240,
        "dumpfile": 240,
        "utl_http": 370,
        "utl_inaddr": 260,
        "dbms_ldap": 380,
        "dbms_xmlquery": 220,
        "xp_cmdshell": 450,
        "sp_configure": 350,
        "sp_oacreate": 340,
        "sp_oamethod": 340,
        "waitfor delay": 260,
        "sleep(": 260,
        "pg_sleep(": 260,
        "benchmark(": 240,
        "extractvalue(": 220,
        "updatexml(": 220,
        "ora_hash(": 150,
        "ascii(": 50,                      # Reduced from 130
        "substring(": 50,                  # Reduced from 120
        "substr(": 50,                     # Reduced from 120
        "mid(": 50,                        # Reduced from 120
        "cast(": 40,                       # Reduced from 100
        "convert(": 40,                    # Reduced from 100
        "char(": 50,                       # Reduced from 140
        "chr(": 50,                        # Reduced from 140
        "0x[0-9a-f]+": 160,
        "/*!": 130,
        "*/": 100,
        "--": 80,
        "#": 80,
        ";": 30,                            # Reduced from 70
        "' or '1'='1": 150,
        "\" or \"1\"=\"1": 150,
        "' or 1=1": 150,
        "\" or 1=1": 150,
        "' and 1=1": 80,                    # Often benign, but part of boolean blind
        "' and 1=2": 100,                   # Boolean blind probe
        "like '%%'": 90,                    # Can be used in blind SQLi

        # --- Cross-Site Scripting (XSS) ---
        "<script": 180,
        "</script>": 100,
        "javascript:": 150,
        "vbscript:": 160,
        "livescript:": 160,
        "data:": 190,
        "alert(": 100,
        "confirm(": 110,
        "prompt(": 110,
        "eval(": 200,
        "document.cookie": 180,
        "document.domain": 150,
        "window.location": 160,
        "location.href": 160,
        "document.write(": 140,
        "innerhtml": 170,
        "outerhtml": 170,
        "setattribute(": 160,
        "onerror=": 200,
        "onload=": 180,
        "onmouseover=": 150,
        "onclick=": 140,
        "onfocus=": 140,
        "onblur=": 140,
        "oninput=": 140,
        "onchange=": 140,
        "onunload=": 170,
        "onabort=": 150,
        "onkeydown=": 150,
        "onkeypress=": 150,
        "onkeyup=": 150,
        "onsubmit=": 190,
        "onreset=": 180,
        "onselect=": 150,
        "<img": 120,
        "src=x": 40,                       # Reduced from 130
        "href=": 30,                       # Reduced from 100
        "action=": 50,                     # Reduced from 150
        "formaction=": 170,
        "<iframe": 170,
        "srcdoc=": 190,
        "<svg": 160,
        "<body": 130,
        "<style": 140,
        "<link": 140,
        "<meta": 150,
        "<object": 180,
        "<embed": 180,
        "<applet": 180,
        "<video": 140,
        "<audio": 140,
        "expression(": 250,
        "&#": 90,
        "String.fromCharCode(": 180,
        "fromcharcode(": 180,
        "constructor": 250,
        "prototype": 270,
        "__proto__": 270,

        # --- Command Injection ---
        "&&": 100,
        "||": 100,
        "|": 120,
        ";": 90,                            # Note: Higher weight here than SQL ; due to context
        "\n": 80,
        "\r": 80,
        "`": 150,
        "$(": 150,
        "${": 140,
        "wget ": 200,
        "curl ": 200,
        "fetch ": 180,
        "nc ": 350,
        "netcat ": 350,
        "bash ": 220,
        "sh ": 210,
        "zsh ": 210,
        "python ": 200,
        "perl ": 200,
        "ruby ": 200,
        "php ": 200,
        "powershell": 220,
        "cmd.exe": 210,
        "/bin/bash": 230,
        "/bin/sh": 220,
        "cat ": 180,
        "type ": 50,                       # Reduced from 180 (common word)
        "ls ": 150,
        "dir ": 150,
        "whoami": 170,
        "id": 50,                           # Reduced from 170 (common word/param)
        "uname ": 160,
        "ifconfig": 170,
        "ip addr": 170,
        "ipconfig": 170,
        "netstat": 180,
        "ss ": 180,
        "route ": 160,
        "ping ": 140,
        "nslookup ": 150,
        "dig ": 150,
        "rm ": 400,
        "del ": 390,
        "mkfifo ": 200,
        "chmod ": 180,
        "chown ": 180,
        "iptables": 220,
        "telnet ": 190,
        "ssh ": 190,
        "scp ": 200,
        "ftp ": 180,
        "> /dev/tcp/": 380,
        "exec ": 240,
        "system(": 350,
        "passthru(": 350,
        "shell_exec(": 350,
        "popen(": 240,
        "proc_open(": 240,

        # --- Local/Remote File Inclusion (LFI/RFI) ---
        "../": 150,
        "..;": 160,
        "..\\": 150,
        "..%00/": 200,
        "/etc/passwd": 200,
        "/etc/shadow": 380,
        "c:\\windows\\system32\\drivers\\etc\\hosts": 180,
        "php://filter": 380,
        "php://input": 250,
        "php://fd/": 240,
        "php://memory": 230,
        "php://temp": 230,
        "zip://": 220,
        "phar://": 230,
        "expect://": 360,
        "data:text/plain": 180,
        "data:application/x-httpd-php": 210,
        "http://": 30,                     # Reduced from 80
        "https://": 30,                    # Reduced from 80
        "ftp://": 50,                      # Reduced from 100
        "require(": 100,
        "include(": 100,
        "require_once(": 100,
        "include_once(": 100,
        "/var/log/": 120,
        "access.log": 100,
        "error.log": 100,
        "apache2/": 110,
        "nginx/": 110,
        "/proc/self/environ": 360,
        "session": 50,                      # Reduced from 100

        # --- Server-Side Template Injection (SSTI) ---
        "{{": 150,
        "}}": 150,
        "{%": 150,
        "%}": 150,
        "${": 160,
        "#{": 160,
        "<%= ": 170,
        "<% ": 160,
        "<%=": 170,
        "<%#": 150,
        "<%__": 150,
        "<?php": 100,
        "<#": 160,
        "[#": 160,
        "[*": 160,
        "*]": 160,
        "$eval": 220,
        "$apply": 200,
        "config.items": 250,
        "__class__": 220,
        "__mro__": 230,
        "__bases__": 230,
        "__subclasses__": 360,
        "__init__": 200,
        "__globals__": 340,
        "__builtins__": 350,
        "__import__": 370,
        "os.system": 380,
        "subprocess.call": 380,
        "java.lang.runtime": 390,
        "freemarker.template.utility.Execute": 400,
        ".tpl.php": 180,
        "self": 180,
        "request": 170,
        "session": 170,
        "application": 170,
        "settings": 170,
        "url_for": 160,
        "getattribute": 200,
        "getitem": 200,
        "popen": 260,

        # --- XML External Entity (XXE) ---
        "<!entity": 400,
        "<!doctype": 100,
        "system ": 350,
        "public ": 200,
        "file:///": 260,
        "http://": 150,
        "ftp://": 160,
        "parameterentity": 380,
        "dtd": 150,
        "xxe": 200,

        # --- Deserialization ---
        "ysoserial": 400,
        "phpggc": 400,
        "rO0": 150,
        "Tzo": 150,
        "PD9waH": 120,
        "AC ED 00 05": 380,
        "aced0005": 380,
        "\xac\xed\x00\x05": 380,
        "java.beans.xmldecoder": 380,
        "objectdataprovider": 380,
        "typeobjectbinder": 270,
        "fastjson": 250,
        "jackson": 200,
        "databind": 210,
        "xstream": 240,
        "pickle": 260,
        "cpickle": 260,
        "__reduce__": 370,
        "yaml.load": 250,
        "!!python": 370,

        # --- NoSQL Injection ---
        "$where": 250,
        "$ne": 100,
        "$eq": 90,
        "$gt": 100,
        "$gte": 100,
        "$lt": 100,
        "$lte": 100,
        "$in": 120,
        "$nin": 120,
        "$regex": 180,
        "$options": 170,
        "$exists": 110,
        "$mod": 130,
        "$text": 140,
        "$search": 140,
        "db.collection": 180,
        "mapreduce": 220,
        "group": 190,
        "function(": 170,
        "this": 100,
        "sleep(": 200,
        "benchmark(": 200,

        # --- XPath Injection ---
        "' or '1'='1": 160,
        "' or count": 150,
        "' or substring": 150,
        "string()": 140,
        "concat()": 140,
        "following-sibling::": 170,
        "preceding-sibling::": 170,
        "ancestor::": 170,
        "parent::": 160,
        "descendant::": 170,
        "attribute::": 150,
        "document(": 200,

        # --- LDAP Injection ---
        "*()|&": 200,
        "objectclass=": 150,
        "(cn=*)": 120,
        "(uid=*)": 120,
        "(&(objectclass=user)(samaccountname=": 180,

        # --- GraphQL Injection / Introspection ---
        "__schema": 250,
        "__typename": 150,
        "introspectionquery": 240,
        "query ": 80,
        "mutation ": 100,
        "fragment ": 90,
        "directives": 160,

        # --- Other / Misc ---
        "jndi:ldap": 450,
        "jndi:rmi": 450,
        "jndi:dns": 300,
        "%24%7Bjndi:": 450,
        "alg=none": 280,
        "select pg_sleep": 260,
        "select benchmark": 240,
        "etc/passwd": 200,
        "windows/win.ini": 150,
        "win.ini": 140,
        "boot.ini": 160,
        "master..sysdatabases": 210,
        "../" * 6 : 250,
        "SELECT.*FROM.*WHERE": 100,
        "<\\?xml": 80,
        "passwd": 50,                       # Reduced from 100
        "password": 50,                     # Reduced from 100
        "secret": 50,                       # Reduced from 110
        "token": 40,                        # Reduced from 110
        "apikey": 60,                       # Reduced from 120
        "admin": 30,                        # Reduced from 70
        "debug=true": 150,
        "test=true": 80,
        "root": 50,                         # Reduced from 100
    }

    for word, weight in attack_words.items():
        # Use regex search for patterns containing special regex chars, otherwise use simple count
        try:
            if any(c in word for c in '.*+?^$()[]{}|\\'): # Basic check for regex metacharacters
                attack_words_weight += len(re.findall(word, inputs, re.IGNORECASE)) * weight
            else:
                attack_words_weight += inputs.lower().count(word.lower()) * weight
        except re.error:
             # Fallback to simple count if regex is invalid for some reason
             attack_words_weight += inputs.lower().count(word.lower()) * weight

    return attack_words_weight


def calculate_manipulation_weight(payload, headers):
    """
    Calculates the weight based on detected manipulations in payload and headers.

    Args:
        payload (str): The request payload.
        headers (dict): The request headers.

    Returns:
        int: The manipulation weight.
    """
    manipulation_weight = 0
    inputs = payload + " ".join(headers.values())

    # --- Manipulation Patterns ---
    # Patterns indicating data manipulation, type juggling, protocol issues, header forgery, etc.
    # These often require more context (field names, expected types/formats/lengths).

    manipulation_patterns = {
        # --- Type/Format/Length Violations ---
        # More specific numeric checks
        r"\b(id|uid|user_id|product_id|item_id|order_id|quantity|count|amount|price|port|pin|zip_code|year|age|sequence)=\D+\b": 120, # Non-digits in common strictly numeric fields
        r"\b(id|uid|user_id|product_id|item_id|order_id|quantity|count|amount|price|port|pin|zip_code|year|age|sequence)=-?\d{15,}\b": 150, # Very large number (potential overflow/DoS)
        r"\b(phone|mobile|tel)=[^0-9\+\-\(\)\s]+\b": 90, # Invalid chars in common phone fields
        r"\b(is_admin|enabled|active|verified|flag|debug|test)=(?!true|false|1|0|on|off|yes|no|y|n)\w+\b": 140, # Non-boolean value in common boolean fields
        # Email format (stricter, but still imperfect)
        r"\bemail=[^@\s]+@[^@\s]+\.[^@\s\.]{2,}(?!\w)": 40, # Basic structure check (weak)
        r"\bemail=.*\s.*@": 80,                 # Space before @ in email
        r"\bemail=.*@.*\s": 80,                 # Space after @ in email
        # Date format examples (adjust to expected formats)
        r"\b(date|timestamp|created_at|updated_at)=\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z\w+": 70, # Extra chars after ISO8601
        r"\b(date|timestamp|created_at|updated_at)=(?!\d{4}-\d{2}-\d{2})\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\w+": 60, # Non-YYYY-MM-DD format with extra chars
        # Potential UUID format violation
        r"\b(uuid|guid|correlation_id)=(?!([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|{[0-9a-fA-F]{8}-...}))\S+": 100,
        # Very long input (generic placeholder - better implemented with field-specific length checks)
        r"\b(username|name|title|subject)=.{200,}\b": 130, # Exceedingly long string for common fields
        r"\b(comment|description|body|message)=.{10000,}\b": 160, # Exceedingly long text block
        r"&[^=;]*;=" : 180,                      # Parameter Pollution (PHP specific: last param wins, ; becomes _) HPP
        r"&[^=]*=&[^=]*=" : 170,                 # Parameter Pollution (ASP.NET specific: all params combined) HPP

        # --- Unexpected Characters / Encodings ---
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]": 180, # Non-printable ASCII control chars (excluding \t, \n, \r)
        r"%00|%u0000": 200,                      # Null Bytes (often for bypassing filters/terminating strings)
        r"(%25)+": 150,                          # Excessive URL encoding (potential filter bypass)
        r"%u[0-9a-fA-F]{4}": 140,                # Unicode encoding (%uXXXX) - sometimes used for bypasses
        r"\\u[0-9a-fA-F]{4}": 130,               # JSON Unicode encoding (\uXXXX) - less common for attacks, but possible
        r"\\x[0-9a-fA-F]{2}": 130,               # Hex encoding (\xXX)

        # --- HTTP Protocol / Header Issues ---
        r"(\r\n|\r|\n){2,}": 70,                 # Multiple consecutive newlines (smuggling/splitting probe)
        r"^(?!Host:)": 200, # Missing Host header (HTTP/1.1 requires it)
        r"Host:\s*[^a-zA-Z0-9\.\-:]{15,}": 150,       # Invalid characters in Host header value
        r"Host:\s*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|\[[0-9a-fA-F:]+\])\s*:\s*\d+": 60, # Host header is IP (can be legit, but sometimes used in SSRF/cache poisoning)
        r"Transfer-Encoding:\s*(?!chunked)[^\r\n]+": 190, # Non-chunked TE (rarely used legitimately now)
        r"Content-Length:\s*[+-]\d+": 220,          # Invalid Content-Length (negative or with sign)
        # HTTP Request Smuggling (HRS) variations
        r"X-Forwarded-Proto:\s*http\b": 80,          # Request claims to be HTTP via proxy (potential TLS bypass if backend trusts)
        r"X-Forwarded-For:\s*[^0-9a-fA-F\.\s,:]+": 100, # Invalid characters in XFF
        # Common scanner/tool User-Agents - Fixed by moving the (?i) flag to the start
        r"(?i)User-Agent:\s*(sqlmap|nmap|nikto|wappalyzer|burp|acunetix|netsparker|zaproxy|owasp zap|masscan|gobuster|dirb|feroxbuster|wfuzz|nuclei)": 120,
        r"(?i)User-Agent:\s*(python-requests|python-urllib|java/|curl/|wget/)": 50, # Common libraries (can be legit, lower weight)
        # Verb Tampering / Non-standard methods
        r"^(?!GET|POST|HEAD|PUT|DELETE|OPTIONS|PATCH|TRACE|CONNECT)\s*[A-Z]+\s+/": 180,
        # Header Injection (CRLF Injection in headers) - Fixed by moving the (?i) flag to the start
        r"(?i)(Set-Cookie|Location|Referer|User-Agent|Content-Type|Authorization|X-[a-zA-Z0-9\-]+):\s*[^\r\n]*(\r|\n|%0d|%0a|%0D|%0A)": 260,
        # Weird spacing/tabs around critical header components
        r"Content-Length\s+:\s*\d+": 90,       # Space before colon
        r"Host\s+:\s*\S+": 90,                 # Space before colon
        r"\r\n\s+": 110,                       # Header folding/obfuscation with whitespace
        # Range requests (potential DoS if mishandled)
        r"Range:\s*bytes=(\d+-\d+,){10,}": 180, # Excessive number of ranges
        r"Range:\s*bytes=(-\d+|\d+-)$": 90,     # Open-ended range (less common, might indicate probing)
        # Content-Type issues
        r"Content-Type:\s*application/xml\b.*<\?php": 220, # XML type with PHP tags inside (XXE/RCE attempt)
        r"Content-Type:\s*multipart/form-data;.*filename=\".*(\.\.|/|\\)\"": 190, # Directory traversal in filename (form upload)
        r"Content-Type:\s*application/x-www-form-urlencoded\b.*\<\w+": 150, # HTML tags in form data (potential XSS bypassing sanitizer expecting plain text)
        r"Content-Type:\s*application/json\b.*[<>]": 160, # HTML tags in JSON (potential XSS)
        # Cache Poisoning Headers
        r"X-Original-URL:": 170,
        r"X-Rewrite-URL:": 170,
        r"X-Forwarded-Host:": 140,
        r"X-Host:": 140,
        r"X-HTTP-Method-Override:": 160,
    } 

    for pattern, weight in manipulation_patterns.items():
        if re.search(pattern, inputs, re.IGNORECASE):
            manipulation_weight += weight
    return manipulation_weight


def calculate_ratio_weight(payload):
    """
    Calculates the alphanumeric to special character ratio weight.

    Args:
        payload (str): The request payload.

    Returns:
        int: The ratio weight.
    """
    payload_length = len(payload)
    if payload_length == 0:
        return 0

    alphanumeric_count = sum(1 for char in payload if char.isalnum())
    special_count = payload_length - alphanumeric_count

# If there are almost no special characters, it's probably not an attack
    if special_count <= 3:
        return 0

    # Calculate the ratio of alphanumeric to special characters
    if special_count == 0:
        ratio = float('inf')  # Avoid division by zero
    else:
        ratio = alphanumeric_count / special_count

    # Normal web traffic typically has a higher ratio of alphanumeric to special chars
    # Lower ratio (more special chars) is more suspicious
    if ratio < 2.0:
        # Higher weight for very low ratios (lots of special characters)
        if ratio < 1.0:
            return 400
        return 200
    else:
    # Normal traffic typically has higher ratios
        return 0


def calculate_files_weight(files):
    """
    Calculates the weight based on suspicious attributes of uploaded files.

    Args:
        files (list): A list of file paths or file objects.

    Returns:
        int: The files weight.
    """
    files_weight = 0

    # Placeholder for invalid file extensions and antivirus scanning results
    invalid_extensions = [".exe", ".bin", ".php"]

    # Placeholder for antivirus detection
    def is_virus_detected(file_path):
        # This is a dummy function - replace with actual antivirus integration
        return False

    for file_path in files:
        file_weight = 0
        if file_path: # Basic check if file information exists
            file_name_parts = file_path.split('.')
            if len(file_name_parts) > 1:
                extension = "." + file_name_parts[-1]
                if extension.lower() in invalid_extensions:
                    file_weight += 500  # Increased weight from 300

            # Dummy antivirus check - replace with actual integration
            if is_virus_detected(file_path):
                file_weight += 200  # For each antivirus (Kaspersky, MalwareBytes, BitDefender)

        files_weight += file_weight
    return files_weight


def parse_csv_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse the CSIC ECML dataset from a CSV file and return a list of request dictionaries
    in the format expected by calculate_attack_weight().

    Args:
        file_path: Path to the CSV file

    Returns:
        List of request dictionaries with the required fields (url, payload, headers, files)
    """
    requests = []

    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Extract headers from the CSV row
                headers = {
                    # 'Cookie': row.get('Cookie', ''),
                    'Host': row.get('Host-Header', ''),
                    'User-Agent': row.get('User-Agent', '')
                }

                # Construct the URL from the URI and host
                method = row.get('Method', '')
                uri = row.get('URI', '')
                host = row.get('Host', 'localhost:8080')
                url = f"http://{host}{uri}"

                # Combine POST data and GET query for payload
                post_data = row.get('POST-Data', '')
                get_query = row.get('GET-Query', '')
                payload = post_data if post_data else get_query
                
                # Create the request dictionary with the expected fields
                request = {
                    'url': url,
                    'payload': payload,
                    'headers': headers,
                    'files': [],  # No files in this dataset
                    'class': row.get('Class', ''),  # Keep original class for reference
                    'method': method,  # Keep method for reference
                    'uri': uri  # Keep URI for reference
                }
                requests.append(request)

    except Exception as e:
        print(f"Error parsing CSV file: {e}")

    return requests


def detect_attacks(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyze a list of requests and detect potential attacks by calculating attack weights.

    Args:
        requests: List of request dictionaries

    Returns:
        List of requests with attack weights added
    """
    results = []

    for req in requests:
        weight = calculate_attack_weight(req)
        req['attack_weight'] = weight
        # Adjusted threshold
        req['is_attack'] = weight > 500  # Threshold for attack classification
        results.append(req)
        
    return results


if __name__ == "__main__":
    # Path to the dataset
    dataset_path = "e:\\FOSS\\langwaf\\data\\csic_ecml_final.csv"

    # Parse the dataset
    print(f"Parsing dataset from {dataset_path}...")
    requests = parse_csv_dataset(dataset_path)
    print(f"Parsed {len(requests)} requests from the dataset")

    for i, request in enumerate(requests[:10]):
        weight = calculate_attack_weight(request)
        print(f"Request {i+1} ({request['method']} {request['uri']}): Attack weight = {weight}")
        print(f"  Class: {request['class']}, Is Attack: {weight > 300}")
        print()
