# Android PackageManager API - Complete Reference

All information obtainable without parsing or opening the APK.

## 1. PACKAGEINFO

Basic package information obtained via: `pm.getPackageInfo(packageName, flags)`

### Always Available
- `packageName` - Package identifier
- `versionName` - Version string (e.g., 1.0.0)
- `versionCode` - Version number (int)
- `longVersionCode` - Version number (long, API 28+)
- `firstInstallTime` - Installation timestamp (milliseconds)
- `lastUpdateTime` - Last update timestamp (milliseconds)

### With GET_PERMISSIONS
- `requestedPermissions[]` - All `<uses-permission>` tags
- `requestedPermissionsFlags[]` - Grant status for each permission

### With GET_META_DATA
- `applicationInfo` - Full application metadata object
- `metaData` - Custom key-value pairs bundle

### With GET_ACTIVITIES
- `activities[]` - All declared activities

### With GET_SERVICES
- `services[]` - All declared services

### With GET_RECEIVERS
- `receivers[]` - All broadcast receivers

### With GET_PROVIDERS
- `providers[]` - All content providers

### With GET_SIGNATURES
- `signatures[]` - Digital signatures of APK

### With GET_GIDS
- `gids[]` - Kernel group IDs

### With GET_CONFIGURATIONS
- `configPreferences[]` - Device configurations

### With GET_SPLIT_NAMES
- `splitNames[]` - Split APK names

---

## 2. APPLICATIONINFO

Application-level metadata obtained via: `pm.getApplicationInfo(packageName, flags)`

### Basic Info
- `packageName` - Package identifier
- `uid` - Kernel user ID
- `enabled` - Is app enabled
- `flags` - App behavior flags

### SDK Versions
- `targetSdkVersion` - Target Android API level
- `minSdkVersion` - Minimum Android API level (API 24+)

### File Paths
- `sourceDir` - Path to base APK
- `publicSourceDir` - Public APK parts
- `dataDir` - App persistent data directory
- `splitSourceDirs[]` - Paths to split APKs
- `nativeLibraryDir` - Native library directory

### Process
- `processName` - Default process name

### UI Elements
- `icon` - Icon resource ID
- `logo` - Logo resource ID
- `label` - Label resource ID
- `theme` - Theme resource ID

### Permissions & Backup
- `permission` - Required access permission
- `backupAgentName` - Backup agent class

### Metadata
- `metaData` - Custom key-value bundle

---

## 3. ACTIVITYINFO

Activity component information (from `PackageInfo.activities` with GET_ACTIVITIES)

- `name` - Activity class name
- `icon` - Icon resource ID
- `label` - Label resource ID
- `permission` - Permission required to launch
- `exported` - Other apps can launch
- `launchMode` - Launch mode (standard, singleTop, singleTask, singleInstance)
- `taskAffinity` - Task affinity
- `screenOrientation` - Required screen orientation
- `configChanges` - Configuration changes handled
- `softInputMode` - Keyboard display mode
- `metaData` - Activity-specific metadata

---

## 4. SERVICEINFO

Service component information (from `PackageInfo.services` with GET_SERVICES)

- `name` - Service class name
- `icon` - Icon resource ID
- `label` - Label resource ID
- `permission` - Permission required to start/bind
- `exported` - Other apps can start
- `processName` - Process running service
- `flags` - Service behavior flags
- `metaData` - Service-specific metadata

---

## 5. PROVIDERINFO

Content provider information (from `PackageInfo.providers` with GET_PROVIDERS)

- `name` - Provider class name
- `authority` - Content authority
- `icon` - Icon resource ID
- `label` - Label resource ID
- `readPermission` - Permission to read
- `writePermission` - Permission to write
- `grantUriPermissions` - Can grant URI permissions
- `multiprocess` - Multiple process support
- `initOrder` - Initialization order
- `metaData` - Provider-specific metadata

---

## 6. RECEIVER INFO

Broadcast receiver information (from `PackageInfo.receivers` with GET_RECEIVERS)

- `name` - Receiver class name
- `icon` - Icon resource ID
- `label` - Label resource ID
- `permission` - Permission required to send
- `exported` - Other broadcasts work
- `metaData` - Receiver-specific metadata

---

## 7. PERMISSIONS

### From PackageInfo
- `requestedPermissions[]` - All requested permission strings
- `requestedPermissionsFlags[]` - Grant status flags

**Check if granted:**
```java
boolean isGranted = (flags[i] & PackageInfo.REQUESTED_PERMISSION_GRANTED) != 0;
```

### From PermissionInfo
Obtained via: `pm.getPermissionInfo(permissionName, flags)`

- `name` - Permission identifier
- `group` - Permission group
- `protectionLevel` - Protection level (normal, dangerous, signature)
- `protectionFlags` - Optional protection flags
- `description` - Permission description
- `label` - User-readable label
- `icon` - Icon resource ID

---

## 8. METADATA

Obtained via: `ApplicationInfo.metaData` (with GET_META_DATA)

- Type: `Bundle` - Custom key-value pairs
- Supported types: String, int, boolean, float, color, resource IDs

```java
ApplicationInfo ai = pm.getApplicationInfo(packageName, PackageManager.GET_META_DATA);
Bundle meta = ai.metaData;
String value = meta.getString("key");
int intValue = meta.getInt("key");
boolean boolValue = meta.getBoolean("key");
```

---

## 9. MAIN PACKAGEMANAGER METHODS

- `getPackageInfo(packageName, flags)` - Single package info
- `getApplicationInfo(packageName, flags)` - Single app info
- `getInstalledPackages(flags)` - All packages list
- `getInstalledApplications(flags)` - All apps list
- `queryIntentActivities(intent, flags)` - Activities for intent
- `queryIntentServices(intent, flags)` - Services for intent
- `queryBroadcastReceivers(intent, flags)` - Receivers for intent
- `getPermissionInfo(permissionName, flags)` - Permission details
- `checkPermission(permission, packageName)` - Check if granted
- `checkSignatures(pkg1, pkg2)` - Compare signatures

---

## 10. PACKAGEMANAGER FLAGS

| Flag | Purpose |
|------|---------|
| `GET_ACTIVITIES` | Return activities array |
| `GET_SERVICES` | Return services array |
| `GET_RECEIVERS` | Return receivers array |
| `GET_PROVIDERS` | Return providers array |
| `GET_PERMISSIONS` | Return permissions arrays |
| `GET_META_DATA` | Return metaData bundle |
| `GET_SIGNATURES` | Return signatures array |
| `GET_GIDS` | Return gids array |
| `GET_CONFIGURATIONS` | Return configuration preferences |
| `GET_SPLIT_NAMES` | Return split APK names |

**Combine flags:** `pm.getPackageInfo(pkg, GET_PERMISSIONS | GET_META_DATA)`

---

## 11. IMPORTANT NOTES

- **No APK Parsing:** All data comes from OS cache at install time
- **Runtime Info:** Includes actual permission grants, not just declarations
- **Android 11+:** Requires `QUERY_ALL_PACKAGES` permission or `<queries>` manifest declaration
- **Performance:** Queries are optimized and cached by OS
- **Flags Combination:** Use bitwise OR (`|`) to combine multiple flags

---

## Example Usage

```java
PackageManager pm = context.getPackageManager();

// Get basic info + permissions
PackageInfo pkgInfo = pm.getPackageInfo(
    packageName,
    PackageManager.GET_PERMISSIONS | PackageManager.GET_META_DATA
);

// Access information
String version = pkgInfo.versionName;
String[] permissions = pkgInfo.requestedPermissions;
int[] flags = pkgInfo.requestedPermissionsFlags;

// Check if permission is granted
for (int i = 0; i < permissions.length; i++) {
    boolean granted = (flags[i] & PackageInfo.REQUESTED_PERMISSION_GRANTED) != 0;
    Log.d("Permission", permissions[i] + ": " + granted);
}

// Access metadata
Bundle metaData = pkgInfo.applicationInfo.metaData;
if (metaData != null) {
    String customValue = metaData.getString("custom_key");
}
```
