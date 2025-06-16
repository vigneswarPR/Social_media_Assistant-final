import requests
from config import Config
import json

def get_page_access_token():
    """Get a Page Access Token from the User Access Token"""
    try:
        # Get the list of pages the user manages
        response = requests.get(
            f"https://graph.facebook.com/v18.0/me/accounts",
            params={
                "access_token": Config.INSTAGRAM_ACCESS_TOKEN,
                "fields": "access_token,name,id"
            }
        )
        
        if response.ok:
            pages = response.json().get('data', [])
            for page in pages:
                if page['id'] == Config.FACEBOOK_PAGE_ID:
                    return page['access_token']
            print(f"❌ Page ID {Config.FACEBOOK_PAGE_ID} not found in managed pages")
            print("Available pages:")
            for page in pages:
                print(f"- {page['name']} (ID: {page['id']})")
            return None
        else:
            print("❌ Error getting pages:", response.text)
            return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def verify_permissions():
    """Verify Facebook/Instagram permissions and display detailed information"""
    if not Config.INSTAGRAM_ACCESS_TOKEN:
        print("❌ Error: INSTAGRAM_ACCESS_TOKEN not found in .env file")
        return

    print("\n🔑 Checking Access Tokens:")
    print("-" * 50)
    
    # Get Page Access Token
    print("\nGetting Page Access Token...")
    page_token = get_page_access_token()
    if page_token:
        print("✅ Successfully got Page Access Token")
    else:
        print("❌ Failed to get Page Access Token")
        print("This is likely why you're seeing the permission error")
        print("\nTo fix this:")
        print("1. Go to https://developers.facebook.com/tools/explorer/")
        print("2. Select your App from the dropdown")
        print("3. Click 'Get User Access Token'")
        print("4. In the permissions popup, select these permissions:")
        print("   - pages_manage_posts")
        print("   - pages_read_engagement")
        print("   - pages_show_list")
        print("   - instagram_basic")
        print("   - instagram_content_publish")
        print("5. Click 'Generate Access Token'")
        print("6. Copy the new token and update your .env file:")
        print("   INSTAGRAM_ACCESS_TOKEN=your_new_token")
        return

    # 1. Verify access token debug information
    debug_url = f"https://graph.facebook.com/debug_token"
    params = {
        "input_token": Config.INSTAGRAM_ACCESS_TOKEN,
        "access_token": f"{Config.FACEBOOK_APP_ID}|{Config.FACEBOOK_APP_SECRET}"
    }
    
    try:
        response = requests.get(debug_url, params=params)
        token_info = response.json()
        
        print("\n🔍 Token Information:")
        print("-" * 50)
        if 'data' in token_info:
            data = token_info['data']
            print(f"Token Valid: {'✅ Yes' if data.get('is_valid', False) else '❌ No'}")
            print(f"Expires: {data.get('expires_at', 'N/A')}")
            print(f"App ID: {data.get('app_id', 'N/A')}")
            print(f"User ID: {data.get('user_id', 'N/A')}")
            
            if 'scopes' in data:
                print("\n📋 Permissions granted:")
                for permission in data['scopes']:
                    print(f"✓ {permission}")
        else:
            print("❌ Error getting token information")
            print(json.dumps(token_info, indent=2))
    
    except Exception as e:
        print(f"❌ Error verifying token: {str(e)}")

    # 2. Verify Page Access
    if Config.FACEBOOK_PAGE_ID:
        try:
            # Try with Page Access Token first
            page_url = f"https://graph.facebook.com/v18.0/{Config.FACEBOOK_PAGE_ID}"
            params = {
                'access_token': page_token,
                'fields': 'name,id,instagram_business_account,roles'
            }
            response = requests.get(page_url, params=params)
            page_info = response.json()
            
            print("\n📱 Facebook Page Information:")
            print("-" * 50)
            if 'error' not in page_info:
                print(f"Page Name: {page_info.get('name', 'N/A')}")
                print(f"Page ID: {page_info.get('id', 'N/A')}")
                if 'instagram_business_account' in page_info:
                    print("✅ Instagram Business Account connected")
                    print(f"Instagram Business ID: {page_info['instagram_business_account']['id']}")
                else:
                    print("❌ No Instagram Business Account connected")
                
                # Check page roles
                if 'roles' in page_info:
                    print("\n👥 Page Roles:")
                    for role in page_info['roles'].get('data', []):
                        print(f"- {role.get('role', 'N/A')}: User {role.get('id', 'N/A')}")
            else:
                print("❌ Error accessing page:")
                print(json.dumps(page_info['error'], indent=2))
        
        except Exception as e:
            print(f"❌ Error verifying page access: {str(e)}")

    # 3. Test posting permission
    try:
        test_url = f"https://graph.facebook.com/v18.0/{Config.FACEBOOK_PAGE_ID}/photos"
        params = {
            "access_token": page_token,  # Use Page Access Token
            "published": "false"  # Don't actually publish
        }
        response = requests.post(test_url, params=params)
        result = response.json()
        
        print("\n📝 Posting Permission Test:")
        print("-" * 50)
        if 'error' in result:
            print("❌ Posting permission test failed:")
            print(f"Error: {result['error'].get('message', 'Unknown error')}")
            print("\n🔧 Required Permissions:")
            print("- pages_manage_posts")
            print("- pages_read_engagement")
            print("- instagram_content_publish")
            print("\nPlease ensure these permissions are granted in your Facebook App settings")
        else:
            print("✅ Posting permission test passed")
    
    except Exception as e:
        print(f"❌ Error testing posting permission: {str(e)}")

if __name__ == "__main__":
    print("🔒 Facebook/Instagram Permission Verification Tool")
    print("=" * 50)
    verify_permissions() 