"""
Simple script to check if database is working
Just run: python check_database.py
"""

import os
import sys

def check_database():
    print("\n" + "="*60)
    print("🔍 CHECKING DATABASE CONNECTION")
    print("="*60 + "\n")
    
    # Step 1: Check if flask app can be imported
    print("Step 1: Importing Flask app...")
    try:
        from flask_api_backend import app, db, Optimization
        print("✅ Flask app imported successfully\n")
    except ImportError as e:
        print(f"❌ Failed to import Flask app: {e}")
        print("   Make sure you're in the project directory\n")
        return False
    
    # Step 2: Check DATABASE_URL
    print("Step 2: Checking DATABASE_URL...")
    db_url = app.config.get('SQLALCHEMY_DATABASE_URI', 'Not configured')
    print(f"   DATABASE_URL: {db_url}")
    
    if 'sqlite' in db_url.lower():
        print("   Type: SQLite (Local database)")
        # Check if file exists
        db_file = db_url.replace('sqlite:///', '')
        if os.path.exists(db_file):
            size = os.path.getsize(db_file)
            print(f"   ✅ Database file exists ({size} bytes)")
        else:
            print("   ⚠️  Database file doesn't exist yet (will be created)")
    elif 'postgresql' in db_url.lower():
        print("   Type: PostgreSQL (Production database)")
    else:
        print("   ⚠️  Unknown database type")
    print()
    
    # Step 3: Test connection
    print("Step 3: Testing database connection...")
    with app.app_context():
        try:
            # Try to execute a simple query
            result = db.session.execute(db.text('SELECT 1'))
            print("✅ Database connection successful!\n")
        except Exception as e:
            print(f"❌ Database connection failed: {e}\n")
            return False
    
    # Step 4: Check tables
    print("Step 4: Checking database tables...")
    with app.app_context():
        try:
            # Get table names
            inspector = db.inspect(db.engine)
            tables = inspector.get_table_names()
            
            if tables:
                print(f"   Found {len(tables)} table(s): {', '.join(tables)}")
                if 'optimizations' in tables:
                    print("   ✅ 'optimizations' table exists")
                else:
                    print("   ⚠️  'optimizations' table not found")
                    print("      Creating tables now...")
                    db.create_all()
                    print("   ✅ Tables created successfully")
            else:
                print("   No tables found. Creating tables...")
                db.create_all()
                print("   ✅ Tables created successfully")
            print()
        except Exception as e:
            print(f"❌ Error checking tables: {e}\n")
            return False
    
    # Step 5: Test read/write
    print("Step 5: Testing read/write operations...")
    with app.app_context():
        try:
            # Count existing records
            count_before = Optimization.query.count()
            print(f"   Current records in database: {count_before}")
            
            # Try to insert a test record
            test_record = Optimization(
                product_id='__test_connection__',
                cost_price=100.0,
                recommended_price=150.0,
                expected_profit=500.0
            )
            db.session.add(test_record)
            db.session.commit()
            print("   ✅ Write operation successful")
            
            # Try to read it back
            found = Optimization.query.filter_by(
                product_id='__test_connection__'
            ).first()
            
            if found:
                print("   ✅ Read operation successful")
                
                # Delete test record
                db.session.delete(found)
                db.session.commit()
                print("   ✅ Delete operation successful")
            else:
                print("   ⚠️  Could not read back test record")
            
            print()
            
        except Exception as e:
            print(f"❌ Read/write test failed: {e}\n")
            db.session.rollback()
            return False
    
    # Step 6: Show some stats
    print("Step 6: Database statistics...")
    with app.app_context():
        try:
            total = Optimization.query.count()
            print(f"   Total optimizations: {total}")
            
            if total > 0:
                recent = Optimization.query.order_by(
                    Optimization.timestamp.desc()
                ).first()
                print(f"   Latest optimization:")
                print(f"      Product: {recent.product_id}")
                print(f"      Price: ${recent.recommended_price:.2f}")
                print(f"      Profit: ${recent.expected_profit:.2f}")
                print(f"      Date: {recent.timestamp}")
            else:
                print("   No optimizations in database yet")
            
            print()
            
        except Exception as e:
            print(f"⚠️  Could not get statistics: {e}\n")
    
    # Final result
    print("="*60)
    print("🎉 DATABASE CHECK PASSED!")
    print("="*60)
    print("\nYour database is working correctly!")
    print("\nNext steps:")
    print("  1. Run your Flask app: python flask-api-backend.py")
    print("  2. Test the API: curl http://localhost:5000/api/health")
    print("  3. View dashboard: http://localhost:5000/admin/dashboard")
    print()
    
    return True


if __name__ == '__main__':
    try:
        success = check_database()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)