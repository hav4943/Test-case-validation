import cantools
import can
import database
from can import message

def read_asc_file(asc_file_path, db):
    # Open the .asc file and process it
    with open(asc_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            print(parts)
            if len(parts) >= 5:
                # Extract timestamp, channel, ID and data from the line
                timestamp = float(parts[0])
                channel = int(parts[1])
                message_id = int(parts[3], 16)  # Hex to int conversion
                data = bytearray.fromhex(''.join(parts[5:]))
 
                # Create a Message object for decoding
                message = message(timestamp=timestamp,
                                  arbitration_id=message_id,
                                  data=data,
                                  is_extended_id=(message_id > 0x7FF))
 
                # Decode the message using the dbc file
                try:
                    decoded_message = db.decode_message(message.arbitration_id, message.data)
                    print(f"Time: {timestamp}, Message ID: {hex(message_id)}, Decoded Data: {decoded_message}")
                except ValueError:
                    # This error occurs if the message ID does not exist in the dbc
                    print(f"Message ID {hex(message_id)} not found in DBC file.")
 
def main():
    # dbc_path =   # Replace with the path to your .dbc file
    # asc_path = '  # Replace with the path to your .asc file
 
    # Load DBC file
    db = cantools.database.load_file('import DBC file path')
    assert isinstance(db, cantools.database.can.database.Database)
 
    # Read and decode .asc file
    read_asc_file('Import .ASC file path', db)
 
if __name__ == "__main__":
    main()
